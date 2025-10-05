# train_misato_full.py
import os, sys, time
import h5py
import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch_geometric.loader import DataLoader
from torch_geometric.nn import NNConv, global_mean_pool

sys.path.insert(0, os.path.join(os.path.abspath('').split('misato-dataset')[0], 'misato-dataset/src/data/components/'))
from data.components.datasets import MolDataset
from data.components.transformQM import GNNTransformQM
from data.processing import preprocessing_db
import torch_geometric.transforms as T

# ---------- paths ----------
qmh5_file = "U:/FYP/misato-dataset/data/QM/h5_files/tiny_qm.hdf5"
norm_file  = "U:/FYP/misato-dataset/data/QM/h5_files/qm_norm.hdf5"
mdh5_file_in = "U:/FYP/misato-dataset/data/MD/h5_files/tiny_md.hdf5"
mdh5_file_out = "U:/FYP/misato-dataset/data/MD/h5_files/tiny_md_out.hdf5"
qm_train_idx = "U:/FYP/misato-dataset/data/QM/splits/train_tinyQM.txt"
qm_val_idx   = "U:/FYP/misato-dataset/data/QM/splits/val_tinyQM.txt"

# ---------- small utilities (edge builder, wrapper, model) ----------
class EdgeBuilderByRadius:
    def __init__(self, radius=4.5, max_nb=32):
        self.r = float(radius); self.max_nb = int(max_nb)
    def __call__(self, data):
        pos = data.pos.cpu().numpy(); n = pos.shape[0]
        rows, cols, e_at = [], [], []
        d = np.linalg.norm(pos[:, None, :] - pos[None, :, :], axis=-1)
        for i in range(n):
            idx = np.where((d[i] <= self.r) & (d[i] > 0.0))[0]
            if idx.size == 0:
                idx = np.argsort(d[i])[1:2]
            else:
                if idx.size > self.max_nb:
                    idx = idx[np.argsort(d[i][idx])[:self.max_nb]]
            for j in idx:
                rows.append(i); cols.append(j); e_at.append([1.0 / (d[i, j] + 1e-8)])
        if len(rows) == 0:
            for i in range(n):
                for j in range(n):
                    if i == j: continue
                    rows.append(i); cols.append(j); e_at.append([1.0 / (np.linalg.norm(pos[i]-pos[j]) + 1e-8)])
        data.edge_index = torch.tensor([rows, cols], dtype=torch.long)
        data.edge_attr  = torch.tensor(e_at, dtype=torch.float)
        return data

class WrapDS(torch.utils.data.Dataset):
    def __init__(self, base_ds, jitter=None, edge_builder=None, y_mean=None, y_std=None):
        self.ds = base_ds; self.jit = jitter; self.eb = edge_builder
        if y_mean is not None:
            self.y_mean = torch.tensor(y_mean, dtype=torch.float)
            self.y_std  = torch.tensor(y_std, dtype=torch.float)
        else:
            self.y_mean = None; self.y_std = None
    def __len__(self): return len(self.ds)
    def __getitem__(self, idx):
        d = self.ds[idx]
        if self.jit is not None: d = self.jit(d)
        if self.eb is not None: d = self.eb(d)
        if (self.y_mean is not None) and (self.y_std is not None):
            y = d.y.float()
            if y.dim() == 1:
                td = self.y_mean.numel()
                if y.numel() == td:
                    d.y = (y - self.y_mean) / (self.y_std + 1e-8)
                elif y.numel() % td == 0:
                    d.y = (y.view(-1, td) - self.y_mean) / (self.y_std + 1e-8)
                else:
                    pass
            else:
                d.y = (y - self.y_mean.to(y.device)) / (self.y_std.to(y.device) + 1e-8)
        return d

def mk_mlp(in_d, out_d, nl=2, hid=64):
    L=[] 
    if nl==1: L.append(nn.Linear(in_d,out_d))
    else:
        L.append(nn.Linear(in_d,hid)); L.append(nn.ReLU())
        for _ in range(nl-2):
            L.append(nn.Linear(hid,hid)); L.append(nn.ReLU())
        L.append(nn.Linear(hid,out_d))
    return nn.Sequential(*L)

class SimpleGNN(nn.Module):
    def __init__(self, in_nf, e_d=1, hid=128, n_steps=3, out_d=1, dropout=0.1):
        super().__init__()
        self.lin0 = nn.Linear(in_nf, hid)
        self.convs = nn.ModuleList()
        for _ in range(n_steps):
            nn_w = mk_mlp(e_d, hid*hid, nl=2, hid=max(hid//4,8))
            self.convs.append(NNConv(hid, hid, nn_w, aggr='mean'))
        self.gru = nn.GRU(hid, hid)
        self.lin1 = nn.Linear(hid, hid//2)
        self.lin2 = nn.Linear(hid//2, out_d)
        self.drop = nn.Dropout(dropout)
    def forward(self, data):
        x = data.x if hasattr(data, 'x') and data.x is not None else data.pos
        x = F.relu(self.lin0(x))
        e = getattr(data, 'edge_attr', None)
        if e is None:
            row, col = data.edge_index
            vec = data.pos[row] - data.pos[col]
            dist = torch.sqrt((vec**2).sum(dim=-1) + 1e-8).unsqueeze(-1)
            e = 1.0/(dist+1e-6)
        h = x.unsqueeze(0)
        for conv in self.convs:
            m = F.relu(conv(x, data.edge_index, e))
            out, h = self.gru(m.unsqueeze(0), h)
            x = out.squeeze(0)
        batch = data.batch if hasattr(data, 'batch') else torch.zeros(x.size(0), dtype=torch.long, device=x.device)
        g = global_mean_pool(x, batch)
        g = self.drop(F.relu(self.lin1(g)))
        out = self.lin2(g)
        if out.dim()==1: out = out.unsqueeze(-1)
        return out

# ---------- helpers ----------
def compute_target_stats(ds, batch_size=16, max_batches=None):
    dl = DataLoader(ds, batch_size=batch_size, shuffle=False, num_workers=0)
    sum_v = None; sum_sq = None; cnt = 0; tgt_dim = None
    for i, b in enumerate(dl):
        y = b.y
        B = int(b.ptr.numel()-1) if hasattr(b, 'ptr') else int(b.batch.max().item()+1)
        if y.dim() == 1:
            if tgt_dim is None:
                assert (y.numel() % B)==0, "can't infer target dim"
                tgt_dim = y.numel() // B
            y = y.view(B, tgt_dim)
        else:
            if tgt_dim is None: tgt_dim = y.shape[1]
        y = y.float()
        if sum_v is None:
            sum_v = y.sum(dim=0); sum_sq = (y**2).sum(dim=0)
        else:
            sum_v += y.sum(dim=0); sum_sq += (y**2).sum(dim=0)
        cnt += y.shape[0]
        if (max_batches is not None) and (i+1 >= max_batches): break
    mean = (sum_v / cnt).cpu().numpy()
    var = (sum_sq / cnt - (sum_v/cnt)**2).cpu().numpy()
    std = np.sqrt(np.maximum(var, 1e-12))
    return mean, std, int(tgt_dim)

def eval_epoch(model, dl, device, mean_t, std_t):
    model.eval()
    loss_fn = nn.MSELoss()
    running = 0.0; n=0
    running_rmse = 0.0
    with torch.no_grad():
        for b in dl:
            b = b.to(device)
            if hasattr(b, 'ptr'): B = int(b.ptr.numel()-1)
            else: B = int(b.batch.max().item()+1)
            y_raw = b.y.float().to(device)
            if y_raw.dim()==1 and y_raw.numel()==B: y = y_raw.view(B,1)
            elif y_raw.dim()==1 and y_raw.numel()==B*mean_t.numel(): y = y_raw.view(B, mean_t.numel())
            else: 
                try:
                    y = y_raw.view(B, -1)
                except:
                    y = y_raw
            y_hat = model(b)
            loss = loss_fn(y_hat, y)
            # unnormalize
            y_hat_unn = y_hat * std_t.unsqueeze(0) + mean_t.unsqueeze(0)
            y_unn = y * std_t.unsqueeze(0) + mean_t.unsqueeze(0)
            rmse = torch.sqrt(((y_hat_unn - y_unn)**2).mean()).item()
            running += float(loss.item()) * B
            running_rmse += rmse * B
            n += B
    return running / max(n,1), running_rmse / max(n,1)

# ---------- training loop ----------
def main():
    torch.manual_seed(42)
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

    # (optional) run preprocessing
    class Args:
        datasetIn = mdh5_file_in; strip_feature="atoms_element"; strip_value=1
        begin=0; end=20; Adaptability=True; Pres_Lat=False; Pocket=0.0; datasetOut = mdh5_file_out
    print("running preprocessing_db.main(...)")
    preprocessing_db.main(Args())

    # base dataset (pass norm file so class doesn't crash)
    base_qm_tr = MolDataset(qmh5_file, qm_train_idx, target_norm_file=norm_file, transform=GNNTransformQM())
    base_qm_val = MolDataset(qmh5_file, qm_val_idx, target_norm_file=norm_file, transform=GNNTransformQM())

    # compute mean/std on training set (if needed)
    mean_vec, std_vec, tgt_dim = compute_target_stats(base_qm_tr, batch_size=16, max_batches=None)
    print("train mean/std:", mean_vec, std_vec, "tgt_dim:", tgt_dim)

    jitter_train = T.RandomJitter(0.25)
    eb = EdgeBuilderByRadius(radius=4.5, max_nb=32)

    train_ds = WrapDS(base_qm_tr, jitter=jitter_train, edge_builder=eb, y_mean=mean_vec, y_std=std_vec)
    val_ds   = WrapDS(base_qm_val, jitter=None, edge_builder=eb, y_mean=mean_vec, y_std=std_vec)  # no jitter on val

    train_dl = DataLoader(train_ds, batch_size=32, shuffle=True, num_workers=0)
    val_dl   = DataLoader(val_ds, batch_size=32, shuffle=False, num_workers=0)

    # infer model dims from a sample
    sample = next(iter(train_dl))
    node_in = sample.x.shape[-1]; e_d = sample.edge_attr.shape[-1]; out_d = int(sample.y.numel() // (sample.ptr.numel()-1) if hasattr(sample,'ptr') else sample.y.shape[1])
    print("node_in", node_in, "edge_d", e_d, "out_d", out_d)

    model = SimpleGNN(in_nf=node_in, e_d=e_d, hid=128, n_steps=3, out_d=out_d).to(device)
    opt = torch.optim.Adam(model.parameters(), lr=1e-3, weight_decay=1e-6)
    sched = torch.optim.lr_scheduler.ReduceLROnPlateau(opt, mode='min', factor=0.5, patience=3, min_lr=1e-5)
    loss_fn = nn.MSELoss()
    mean_t = torch.tensor(mean_vec, dtype=torch.float).to(device)
    std_t  = torch.tensor(std_vec, dtype=torch.float).to(device)

    best_val_rmse = 1e9; best_epoch = -1; patience = 5; wait = 0
    n_epochs = 20

    for epoch in range(1, n_epochs+1):
        t0 = time.time()
        model.train()
        running = 0.0; n_b = 0
        for b in train_dl:
            b = b.to(device)
            if hasattr(b,'ptr'): B = int(b.ptr.numel()-1)
            else: B = int(b.batch.max().item()+1)
            y_raw = b.y.float().to(device)
            # reshape y
            if y_raw.dim()==1 and y_raw.numel()==B*out_d: y = y_raw.view(B, out_d)
            elif y_raw.dim()==2: y = y_raw
            else:
                try: y = y_raw.view(B, -1)
                except: y = y_raw
            opt.zero_grad()
            y_hat = model(b)
            loss = loss_fn(y_hat, y)
            loss.backward()
            torch.nn.utils.clip_grad_norm_(model.parameters(), 5.0)
            opt.step()
            running += float(loss.item()) * B
            n_b += B
        train_loss = running / max(1, n_b)

        # eval
        val_loss, val_rmse = eval_epoch(model, val_dl, device, mean_t, std_t)
        sched.step(val_loss)

        print(f"Epoch {epoch:02d} | train_norm_MSE={train_loss:.6f} | val_norm_MSE={val_loss:.6f} | val_unnorm_RMSE={val_rmse:.6f} | time={time.time()-t0:.1f}s")

        # checkpoint by unnorm RMSE
        if val_rmse < best_val_rmse:
            best_val_rmse = val_rmse; best_epoch = epoch; wait = 0
            torch.save({'epoch': epoch, 'model_state': model.state_dict(), 'opt_state': opt.state_dict()}, "best_model.pth")
            print(" saved best_model.pth")
        else:
            wait += 1
            if wait >= patience:
                print(f"Early stopping after {epoch} epochs (no improvement for {patience} epochs), best epoch {best_epoch} val_rmse={best_val_rmse:.4f}")
                break

    print("Training finished. Best epoch:", best_epoch, "best_val_rmse:", best_val_rmse)

if __name__ == "__main__":
    main()
