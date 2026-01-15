import numpy as np
import torch
import torch.utils.data as Data
import scipy.stats as st

#%% Define the functions for generating data in univariate and multivariate cases


def g1(q):
    q1 = q[:, 0]
    q2 = q[:, 1]
    out1 = (2-2*torch.abs(q1-0.5)) + torch.exp(2* q1 * q2)
    out2 = 2*torch.cos( np.pi * (q2))
    return torch.stack([out1, out2], dim=1)

def g2(q):
    q1 = q[:, 0]
    q2 = q[:, 1]
    return (torch.sqrt(q1 + q2**2) + q1**2 * q2).unsqueeze(dim=1)


def g3(q):
    norm = torch.norm(q-0.5,2,dim=1)
    return norm.unsqueeze(dim=1)

def h1(q):
    q1 = q[:,:5];q2 = q[:,5:8];q3 = q[:,8:10];
    center1 = torch.tensor([0.5]*(q1.shape[1]), dtype=q1.dtype, device=q1.device)
    out1 = 2*torch.norm(q1 - center1, dim=1)
    out2 = 2*torch.cos(np.pi * (torch.sum(q2,dim=1)))
    out3 = 20*torch.prod(q3,dim=1)
    return torch.stack([out1, out2, out3], dim=1)


def h2(q):
    out1 = torch.sqrt(4*q[:,0]+q[:,2])
    out2 = torch.pow(q[:,0],2)+torch.sum(2*(q[:,1:]-1)**2,dim=1)
    return torch.stack([out1, out2], dim=1)


def h3(q):
    out = q[:,1]+torch.sqrt(q[:,0]+torch.abs(q[:,1]))
    return out.unsqueeze(dim=1)



def gen_multi(A=None,B=None,size=2**10,d=8,model='sim',error='t',df=3,sigma=1):
    if A==None:
        torch.manual_seed(2024);A=-1*torch.randint(0,3,[d,1])*torch.randn([d,1]);
    if B==None:
        torch.manual_seed(2025);B=-1*torch.randint(0,3,[d,1])*torch.randn([d,1]);
    x = torch.rand([size,d]).float()
    
    if error == 't':
        eps = torch.from_numpy(np.random.standard_t(df, [size, 1]))
    elif error == 'normal':
        eps = torch.randn([size, 1])
    elif error == 'sinex':
        eps = torch.randn([size, 1]) * (torch.sin(np.pi * torch.from_numpy(np.dot(x.data.numpy(), B.data.numpy()))))
    elif error == 'expx':
        eps = torch.randn([size, 1]) * (torch.exp((torch.from_numpy(np.dot(x.data.numpy(), B.data.numpy())) - 0.5) * 0.1))
    elif error == 'sinex_t':
        eps = torch.from_numpy(np.random.standard_t(df, [size, 1])) * (torch.sin(np.pi * torch.from_numpy(np.dot(x.data.numpy(), B.data.numpy()))))
    elif error == 'expx_t':
        eps = torch.from_numpy(np.random.standard_t(df, [size, 1])) * (torch.exp((torch.from_numpy(np.dot(x.data.numpy(), B.data.numpy())) - 0.5) * 0.1))
    elif error == 'scenario1':
        eps = torch.from_numpy(np.random.standard_t(df, [size, 1])) * g3(x)
    else:
        raise ValueError(f"Unknown error type: {error}")
    eps = eps.float()
    
    
    if model == 'scenario3':
        y = torch.pow(1 * torch.from_numpy(np.dot(x.data.numpy(), A.data.numpy())),2) + sigma * eps
    elif model == 'add':
        y = -3 * x[:, 0].unsqueeze(1) + 4 * torch.pow(x[:, 1].unsqueeze(1) - 0.5, 2) + 5 * torch.sin(np.pi * x[:, 2].unsqueeze(1)) + 6 * torch.abs(x[:, 3].unsqueeze(1) - 0.5) + sigma * eps
    elif model == 'linear':
        y = 2 * torch.from_numpy(np.dot(x.data.numpy(), A.data.numpy())) + sigma * eps
    elif model == 'scenario1':
        y = g2(g1(x))
    elif model == 'scenario2':
        y = h3(h2(h1(x)))
    else:
        raise ValueError(f"Unknown model type: {model}")
    y = y.float()

    u = torch.rand([size,1]).float()
    
    return Data.TensorDataset(x, y, u, eps)


def quant_multi(x,taus,A,B,model='sim',error='t',df=3,sigma=1):
    if x.shape[0]!=taus.shape[0]:
        taus=(taus.T).repeat([x.shape[0],1]);

    if error == 't':
        eps = torch.from_numpy(sigma * st.t.ppf(taus, df=df))
    elif error == 'normal':
        eps = torch.from_numpy(sigma * st.norm.ppf(taus))
    elif error == 'sinex':
        temp = torch.from_numpy(np.dot(x.data.numpy(), B.data.numpy()))
        eps = torch.abs(torch.sin(np.pi * temp)) * sigma * st.norm.ppf(taus)
    elif error == 'expx':
        temp = torch.from_numpy(np.dot(x.data.numpy(), B.data.numpy()))
        eps = torch.exp((temp - 0.5) * 0.1) * sigma * st.norm.ppf(taus)
    elif error == 'sinex_t':
        temp = torch.from_numpy(np.dot(x.data.numpy(), B.data.numpy()))
        eps = torch.abs(torch.sin(np.pi * temp)) * sigma * st.t.ppf(taus, df=df)
    elif error == 'expx_t':
        temp = torch.from_numpy(np.dot(x.data.numpy(), B.data.numpy()))
        eps = torch.exp((temp - 0.5) * 0.1) * sigma * st.t.ppf(taus, df=df)
    elif error == 'scenario1':
        eps = torch.from_numpy(sigma * st.t.ppf(taus, df=df)) * g3(x)
    else:
        raise ValueError(f"Unknown error type: {error}")
    eps = eps.float()
    
    if model == 'scenario3':
        quantile = torch.pow(1 * torch.from_numpy(np.dot(x.data.numpy(), A.data.numpy())),2) + eps
    elif model == 'add':
        quantile = -3 * x[:, 0].unsqueeze(1) + 4 * torch.pow(x[:, 1].unsqueeze(1) - 0.5, 2) + 5 * torch.sin(np.pi * x[:, 2].unsqueeze(1)) + 6 * torch.abs(x[:, 3].unsqueeze(1) - 0.5) + eps
    elif model == 'linear':
        quantile = 2 * torch.from_numpy(np.dot(x.data.numpy(), A.data.numpy())) + eps
    elif model == 'scenario1':
        quantile = g2(g1(x)) + eps
    elif model == 'scenario2':
        quantile = h3(h2(h1(x))) +eps
    else:
        raise ValueError(f"Unknown model type: {model}")
    quantile = quantile.float()

    return quantile




def gen_univ(size=2**10,model='wave',error='expsinex', df=2,sigma=1):
    x = torch.rand([size,1]).float()
    errors={'t':torch.from_numpy(np.random.standard_t(df,[size,1])),
            'normal':torch.randn([size,1]),
            'cauchy':torch.from_numpy(np.random.standard_cauchy([size,1])),
            'sinex':torch.randn(x.shape)*torch.sin(np.pi*x),
            'expx':torch.randn(x.shape)*(torch.exp((x-0.5)*2)),
            'cross': torch.randn(x.shape)*torch.sin(0.99*np.pi*x),
            'sinex_t':torch.from_numpy(np.random.standard_t(df,[size,1]))*torch.sin(np.pi*x),
            'expx_t':torch.from_numpy(np.random.standard_t(df,[size,1]))*(torch.exp((x-0.5)*2)),
            }
    eps=errors[error].float()
    ys={'wave':2*x*torch.sin(4*np.pi*x)+sigma*eps,
         'linear':2*x+sigma*eps,
         'exp': torch.exp(2*x)+sigma*eps,
         'angle': (4-4*torch.abs(x-0.5))+sigma*eps,
         'iso':2*np.pi*x+torch.sin(2*np.pi*x)+sigma*eps,
         'constant': torch.ones_like(x)+sigma*eps,
        }
    
    u=torch.rand([size,1]).float();
    y=ys[model].float()
    return Data.TensorDataset(x, y, u, eps)



def quant_univ(x,taus,model='sine',error='expx',df=2,sigma=1):
    if x.shape[0]!=taus.shape[0]:
        taus=(taus.T).repeat([x.shape[0],1]);
    errors={'t':torch.from_numpy(sigma*st.t.ppf(taus,df=df)),
            'normal':torch.from_numpy(sigma*st.norm.ppf(taus)),
            'sinex':torch.sin(np.pi*x)*sigma*st.norm.ppf(taus),
            'expx':torch.exp((x-0.5)*2)*sigma*st.norm.ppf(taus),
            'cross':torch.sin(0.99*np.pi*x)*sigma*st.norm.ppf(taus),
            'sinex_t':torch.sin(np.pi*x)*sigma*st.t.ppf(taus,df=df),
            'expx_t':torch.exp((x-0.5)*2)*sigma*st.t.ppf(taus,df=df),
            }
    eps=errors[error].float()
    quantiles={'wave':2*x*torch.sin(4*np.pi*x)+eps,
         'linear':2*x+eps,
         'exp': torch.exp(2*x)+eps,
         'angle': (4-4*torch.abs(x-0.5))+eps,
         'iso':2*np.pi*x+torch.sin(2*np.pi*x)+eps,
         'constant': torch.ones_like(x)+eps,
        }
    quantile=quantiles[model].float()
    return quantile
    
    

    