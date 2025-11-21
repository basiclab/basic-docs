---
title: "Introduction to Multi-GPU Training 1: From First Principles to Production"
description: A series of tutorials on the mechanism, method, and best practices of multi-GPU training.
slug: intro-to-multi-gpu-1
tags: [gpu, multi-gpu]
---

:::note
This tutorial assumes you already know the basics of PyTorch and how to train a model.
:::

If you've been training deep learning models on a single GPU and wondering how to scale up, or if you've heard terms like "data parallelism" and "AllReduce" thrown around without really understanding what they mean - this series is for you. We're going to build your understanding from the ground up, starting with the absolute basics of how multiple GPUs can work together, all the way to implementing production-ready distributed training systems.

This first article lays the foundation. We'll understand why we need multiple GPUs, how they actually communicate with each other, and what happens under the hood when you run distributed training. By the end, you'll have the mental model needed to understand everything that comes next in this series.

<!-- truncate -->

## Why Multiple GPUs?

Let's start with a concrete example. Suppose you want to train a 7 billion (7B) parameter language model. Each parameter is typically stored as a 32-bit float (4 bytes). That's:

```plaintext
7B parameters × 4 bytes = 28GB just for the model weights
```

But training requires more than just storing weights. You also need:

- **Gradients**: Another 28GB (one gradient per parameter)
- **Optimizer states**: For Adam, that's 56GB more (two momentum terms per parameter)
- **Activations**: Varies by batch size, but easily another 20-40GB

**Total memory needed: ~130GB+**

Even a high-end NVIDIA H100 GPU with 80GB of memory can't handle this alone. You physically need multiple GPUs just to fit the model in memory.

But there's another reason: time. Training large models on a single GPU can take weeks or months. If you can split the work across multiple GPUs effectively, you can train the same model in days or hours.

The key word here is "effectively" - and that's what this series is all about.

## Understanding the Four Parallelism Strategies

Before we dive deep, let's map out the landscape. There are four main ways to distribute training across multiple GPUs:

1. **Data Parallelism**

```plaintext
GPU 0: [Full Model] processes Batch 0
GPU 1: [Full Model] processes Batch 1
GPU 2: [Full Model] processes Batch 2
GPU 3: [Full Model] processes Batch 3
        ↓ sync gradients ↓
All GPUs update with averaged gradients
```

Each GPU has a complete copy of the model but processes different data. This is the most common approach and what you'll use 80% of the time.

2. **Model Parallelism**

```plaintext
GPU 0: [Layer 1-3]  →  GPU 1: [Layer 4-6]  →  GPU 2: [Layer 7-9]
```

Split the model vertically across layers. Data flows through the pipeline. Used when the model is too large to fit on one GPU.

3. **Tensor Parallelism**

```plaintext
        Input
          ↓
    ┌─────┴─────┐
GPU 0: [Layer 1a] GPU 1: [Layer 1b]
    └─────┬─────┘
       Concat
```

Split individual layers horizontally across GPUs. Each GPU computes part of each layer. Used for very large layers (like in transformers).

4. **Pipeline Parallelism**

```plaintext
Time →
GPU 0: [Micro-batch 1] [Micro-batch 2] [Micro-batch 3]
GPU 1:                 [Micro-batch 1] [Micro-batch 2]
GPU 2:                                 [Micro-batch 1]
```

Like model parallelism but with better GPU utilization through pipelining. Splits model into stages and processes multiple micro-batches simultaneously.

Most production systems use hybrid approaches, combining multiple strategies. But we'll tackle them one at a time in this series.

## How GPUs Actually Communicate

Now here's the crucial part that many tutorials skip: how do GPUs actually talk to each other? Understanding this will help you reason about performance, debug issues, and make better architectural decisions.

### The Hardware: PCIe vs NVLink

GPUs connect to each other (and the CPU) through physical interconnects. The two main types you'll encounter are:

**PCIe (Peripheral Component Interconnect Express)**

This is the standard connection. Think of it like a highway system where all traffic has to go through the CPU as a central hub:

```plaintext
         CPU
          |
    [PCIe Switch]
    /    |    \
 GPU0  GPU1  GPU2
```

- **PCIe 3.0 x16**: ~16 GB/s per direction
- **PCIe 4.0 x16**: ~32 GB/s per direction
- **PCIe 5.0 x16**: ~64 GB/s per direction

The limitation: When GPU 0 wants to send data to GPU 1, it often has to go through the CPU or a PCIe switch first. This adds latency and limits bandwidth.

**NVLink (NVIDIA's High-Speed Interconnect)**

NVLink gives GPUs direct, high-bandwidth connections to each other:

```plaintext
GPU0 ═══════ GPU1
 ║            ║
GPU2 ═══════ GPU3
```

- **NVLink 2.0** (V100): ~300 GB/s bidirectional between GPUs
- **NVLink 3.0** (A100): ~600 GB/s bidirectional
- **NVLink 4.0** (H100): ~900 GB/s bidirectional

That's 20-30x faster than PCIe! This is why high-end training servers use NVLink-connected GPUs. The difference isn't just academic - it directly impacts how fast you can train.

You can check your system's GPU topology with:

```bash
nvidia-smi topo -m
```

Look for "NV" entries (meaning NVLink) vs "PIX" or "PXB" (meaning PCIe). The hardware topology will determine which parallelism strategies work well for you.

For example, here's the topology output from the `basic-3` server:

```plaintext
        GPU0    GPU1    GPU2    GPU3    CPU Affinity    NUMA Affinity
GPU0     X      NV12    PXB     PXB     0-31,64-95           0
GPU1    NV12     X      PXB     PXB     0-31,64-95           0
GPU2    PXB     PXB      X      NV12    0-31,64-95           0
GPU3    PXB     PXB     NV12     X      0-31,64-95           0
```

the annotation means:

- **NV12**: GPUs connected via NVLink (GPU0↔GPU1 and GPU2↔GPU3)
- **PXB**: GPUs connected via PCIe bridge (all other pairs)
- **X**: The GPU compared to itself

This server has a pair topology: GPU0 and GPU1 are tightly coupled via NVLink, as are GPU2 and GPU3. However, communication between pairs (e.g., GPU0 to GPU2) must go through PCIe, which is slower.

### The Software Stack: From Sockets to NCCL

On top of the hardware, we have layers of software that handle communication. Here's the full stack:

```plaintext
┌────────────────────────────────┐
│  Your Training Code            │ ← What you write
├────────────────────────────────┤
│  PyTorch DDP / Accelerate      │ ← High-level frameworks
├────────────────────────────────┤
│  torch.distributed             │ ← PyTorch's abstraction
├────────────────────────────────┤
│  NCCL / Gloo / MPI             │ ← Communication backends
├────────────────────────────────┤
│  TCP/IP / InfiniBand           │ ← Network transport
├────────────────────────────────┤
│  GPU Direct RDMA               │ ← Hardware acceleration
└────────────────────────────────┘
```

- **NCCL (NVIDIA Collective Communications Library)** is the most important one to know. It's specifically optimized for NVIDIA GPUs and knows how to use NVLink, PCIe, and even multi-node networks efficiently. When you use PyTorch's distributed training on NVIDIA GPUs, you're almost certainly using NCCL under the hood.
- **Gloo** is Facebook's communication library. It works on both CPUs and GPUs and is more portable than NCCL, but typically slower for GPU-to-GPU communication.
- **MPI (Message Passing Interface)** is the old-school choice from the high-performance computing world. It's very mature but wasn't designed with GPUs in mind.

### Collective Operations: The Building Blocks

When training distributed models, you don't just send arbitrary messages between GPUs. Instead, you use coordinated communication patterns called "collective operations." Here are the key ones:

- **Broadcast: Send to Everyone**

```plaintext
Before:              After:
GPU 0: [Model]       GPU 0: [Model]
GPU 1: [Empty]  →    GPU 1: [Model]
GPU 2: [Empty]       GPU 2: [Model]
GPU 3: [Empty]       GPU 3: [Model]
```

One GPU (usually rank 0) sends data to all others. Used to distribute the initial model weights.

- **AllReduce: The Star of Distributed Training**

```plaintext
Before:              After (sum):
GPU 0: [grad=1.0]    GPU 0: [grad=10.0]
GPU 1: [grad=2.0] →  GPU 1: [grad=10.0]
GPU 2: [grad=3.0]    GPU 2: [grad=10.0]
GPU 3: [grad=4.0]    GPU 3: [grad=10.0]
```

Every GPU contributes data, an operation is applied (usually sum or average), and every GPU gets the result. This is how gradients are synchronized in data parallel training.

Why is AllReduce so important? In data parallelism:

1. Each GPU computes gradients on its local batch
2. AllReduce sums all gradients across GPUs
3. Each GPU divides by the number of GPUs (to average)
4. Each GPU updates its model with the averaged gradients
5. All models stay synchronized!

The naive way to implement AllReduce would be for every GPU to send to every other GPU (O(N²) communication). But NCCL uses clever algorithms like Ring AllReduce that only require O(N) communication:

```plaintext
Ring AllReduce (simplified):
Step 1: GPU0→GPU1, GPU1→GPU2, GPU2→GPU3, GPU3→GPU0
Step 2: GPU0→GPU1, GPU1→GPU2, GPU2→GPU3, GPU3→GPU0
Step 3: GPU0→GPU1, GPU1→GPU2, GPU2→GPU3, GPU3→GPU0
```

Each GPU sends and receives from just two neighbors, but everyone ends up with the complete result.

- **Gather and Scatter**

```plaintext
Scatter (one-to-many):           Gather (many-to-one):
GPU 0: [A,B,C,D]                 GPU 0: [A]
GPU 1: []        →  Scatter →    GPU 1: [B]
GPU 2: []                        GPU 2: [C]
GPU 3: []                        GPU 3: [D]

                                 ↓ Gather ↓
                                
                                 GPU 0: [A,B,C,D]
```

Scatter distributes different data to different GPUs (used for splitting batches). Gather collects data from all GPUs to one (used for collecting metrics).

### A Simple Example: Understanding Communication Overhead

Let's make this concrete with some real numbers. Suppose you're training a ResNet-50 model (25 million parameters) with data parallelism on 4 GPUs:

**Model size in FP32**: 25M parameters × 4 bytes = 100MB

During each training iteration:

1. Each GPU processes its local batch: ~50ms (forward + backward)
2. AllReduce to synchronize gradients: Depends on hardware!

**With PCIe (16 GB/s):**

```plaintext
Communication time = 100MB / 16 GB/s ≈ 6ms
Iteration time = 50ms + 6ms = 56ms
Communication overhead = 6/56 ≈ 11%
```

**With NVLink (300 GB/s):**

```plaintext
Communication time = 100MB / 300 GB/s ≈ 0.3ms
Iteration time = 50ms + 0.3ms = 50.3ms
Communication overhead = 0.3/50.3 ≈ 0.6%
```

See the difference? With NVLink, communication is nearly free. With PCIe, it eats 11% of your time. For larger models (billions of parameters), this overhead can become much worse with PCIe.

This is why understanding your hardware matters. The same code will perform very differently depending on whether you have NVLink or just PCIe.

## A Hands-On Example: Building Mini-AllReduce

To really understand what's happening, let's build a minimal version of AllReduce using Python sockets. This will show you what PyTorch is doing under the hood (though much more slowly and simply).

Here's a basic server that receives data from multiple clients, sums it, and sends the result back:

<details>
<summary>mini_allreduce_server.py</summary>

```python
import socket
import pickle
import numpy as np

def run_allreduce_server(num_clients: int = 4, port: int = 29500):
    """
    A minimal parameter server implementing AllReduce.
    Receives data from all clients, sums it, and broadcasts back.
    """
    server = socket.socket(socket.AF_INET, socket.SOCK_STREAM)
    server.setsockopt(socket.SOL_SOCKET, socket.SO_REUSEADDR, 1)
    server.bind(("0.0.0.0", port))
    server.listen(10)
    
    print(f"AllReduce Server listening on port {port}")
    print(f"Waiting for {num_clients} clients...\n")
    
    # Accept all clients
    clients = []
    for i in range(num_clients):
        client, addr = server.accept()
        clients.append(client)
        print(f"Client {i} connected")
    
    print("\nStarting AllReduce operation...\n")
    
    # Receive data from all clients
    data_list = []
    for i, client in enumerate(clients):
        # Simple protocol: receive size, then data
        size = int.from_bytes(client.recv(4), byteorder="big")
        data_bytes = client.recv(size)
        data = pickle.loads(data_bytes)
        data_list.append(data)
        print(f"Received from client {i}: {data}")
    
    # Sum all data (the "Reduce" part)
    result = np.sum(data_list, axis=0)
    print(f"\nSum of all data: {result}")
    
    # Send result to all clients (the "All" part)
    result_bytes = pickle.dumps(result)
    for i, client in enumerate(clients):
        client.send(len(result_bytes).to_bytes(4, byteorder="big"))
        client.sendall(result_bytes)
        print(f"Sent result to client {i}")
    
    # Cleanup
    for client in clients:
        client.close()
    server.close()
    print("\nAllReduce completed!")

if __name__ == "__main__":
    run_allreduce_server()
```

</details>

And here's a client that connects and participates:

<details>
<summary>mini_allreduce_client.py</summary>

```python
import socket
import pickle
import numpy as np
import sys

def run_allreduce_client(rank: int, data: np.ndarray, port: int = 29500):
    """
    Connect to server and participate in AllReduce.
    
    Args:
        rank: This client's ID (simulates GPU rank)
        data: The gradient array to reduce
    """
    client = socket.socket(socket.AF_INET, socket.SOCK_STREAM)
    client.connect(("127.0.0.1", port))
    
    print(f"[Rank {rank}] Connected to server")
    print(f"[Rank {rank}] Sending data: {data}")
    
    # Send data
    data_bytes = pickle.dumps(data)
    client.send(len(data_bytes).to_bytes(4, byteorder="big"))
    client.sendall(data_bytes)
    
    # Receive result
    size = int.from_bytes(client.recv(4), byteorder="big")
    result_bytes = client.recv(size)
    result = pickle.loads(result_bytes)
    
    print(f"[Rank {rank}] Received result: {result}")
    
    client.close()
    return result

if __name__ == "__main__":
    if len(sys.argv) < 2:
        print("Usage: python mini_allreduce_client.py <rank>")
        sys.exit(1)
    
    rank = int(sys.argv[1])
    
    # Each client sends different data (simulating different gradients)
    data = np.array([1.0, 2.0, 3.0]) * (rank + 1)
    
    result = run_allreduce_client(rank, data)
    
    # Verify: sum should be [10.0, 20.0, 30.0]
    expected = np.array([10.0, 20.0, 30.0])
    print(f"[Rank {rank}] Correct: {np.allclose(result, expected)}")
```

</details>

**Try it yourself:**

```bash
# Terminal 1: Start the server
python mini_allreduce_server.py

# Terminals 2-5: Start clients (simulating 4 GPUs)
python mini_allreduce_client.py 0
python mini_allreduce_client.py 1
python mini_allreduce_client.py 2
python mini_allreduce_client.py 3
```

You'll see:

```plaintext
Server output:
Received from client 0: [1. 2. 3.]
Received from client 1: [2. 4. 6.]
Received from client 2: [3. 6. 9.]
Received from client 3: [4. 8. 12.]
Sum of all data: [10. 20. 30.]

Each client output:
[Rank 0] Received result: [10. 20. 30.]
```

This is exactly what happens during distributed training! Each "client" represents a GPU that:

1. Computes local gradients (the input data)
2. Participates in AllReduce (sends to server, receives sum)
3. Uses the averaged result to update parameters

**Wait, but who is the "server" in real multi-GPU training?**

Great question! In our simple example above, we used a centralized server for clarity. But in real distributed training with libraries like NCCL, there's actually **no dedicated server process**. Instead:

- **Every GPU is both a client and a server** - they all participate as peers
- Communication happens **peer-to-peer** using algorithms like Ring AllReduce
- Data flows in patterns (rings, trees, etc.) without a central coordinator
- This is called **decentralized** or **peer-to-peer communication**

Here's the difference:

```
Our Simple Example (Centralized):
GPU0 ──┐
GPU1 ──┼──→ [Server] ──→ Sum ──→ Broadcast to all
GPU2 ──┤
GPU3 ──┘

Real NCCL (Decentralized Ring):
GPU0 ⇄ GPU1
 ⇅       ⇅
GPU3 ⇄ GPU2
(Each GPU sends to its neighbor in a ring)
```

The centralized approach is easier to understand, but the decentralized approach is much more efficient because:

- No single bottleneck (the server)
- Better bandwidth utilization (all GPUs communicate simultaneously)
- Scales better to hundreds or thousands of GPUs

Of course, real AllReduce is much more sophisticated:

- Uses GPU-optimized ring or tree algorithms
- Leverages GPU Direct RDMA to bypass the CPU entirely
- Overlaps communication with computation
- Handles network failures gracefully

But the core concept is identical.

## Distributed Training with PyTorch

Now that you understand the fundamentals, let's actually implement distributed training. The best way to understand multi-GPU training is to start with single GPU code, then see exactly what changes.

We prepare a simple GPT-style language model for demonstration and the dataset.

<details>
<summary>model.py</summary>

```python
import torch
import torch.nn as nn

class SimpleGPT(nn.Module):
    """
    A minimal GPT-style language model for demonstration.
    GPT uses a decoder-only transformer architecture with causal masking.
    """
    def __init__(self, vocab_size: int = 50257, d_model: int = 512, n_heads: int = 8, n_layers: int = 6, max_seq_len: int = 512):
        super().__init__()
        
        self.d_model = d_model
        self.vocab_size = vocab_size
        
        # Embeddings
        self.token_embedding = nn.Embedding(vocab_size, d_model)
        self.position_embedding = nn.Embedding(max_seq_len, d_model)
        
        # Transformer decoder blocks (GPT is decoder-only)
        decoder_layer = nn.TransformerDecoderLayer(
            d_model=d_model,
            nhead=n_heads,
            dim_feedforward=d_model * 4,
            dropout=0.1,
            batch_first=True,
            norm_first=True  # Pre-LN like modern transformers
        )
        self.transformer = nn.TransformerDecoder(decoder_layer, num_layers=n_layers)
        
        # Final layer norm and output head
        self.ln_f = nn.LayerNorm(d_model)
        self.head = nn.Linear(d_model, vocab_size, bias=False)
        
        # Weight tying (share weights between token embedding and output head)
        self.head.weight = self.token_embedding.weight
        
        # Initialize weights
        self.apply(self._init_weights)
        
        print(f"Model initialized with {self.count_parameters()/1e6:.2f}M parameters")
    
    def _init_weights(self, module: nn.Module):
        """Initialize weights following GPT-2 initialization."""
        if isinstance(module, nn.Linear):
            torch.nn.init.normal_(module.weight, mean=0.0, std=0.02)
            if module.bias is not None:
                torch.nn.init.zeros_(module.bias)
        elif isinstance(module, nn.Embedding):
            torch.nn.init.normal_(module.weight, mean=0.0, std=0.02)
    
    def count_parameters(self) -> int:
        """Count trainable parameters."""
        return sum(p.numel() for p in self.parameters() if p.requires_grad)
    
    def forward(self, input_ids: torch.Tensor) -> torch.Tensor:
        """
        Forward pass.
        
        Args:
            input_ids: Token IDs of shape (batch_size, seq_len)
            
        Returns:
            logits: Output logits of shape (batch_size, seq_len, vocab_size)
        """
        batch_size, seq_len = input_ids.shape
        device = input_ids.device
        
        # Create position indices [0, 1, 2, ..., seq_len-1]
        positions = torch.arange(0, seq_len, dtype=torch.long, device=device)
        positions = positions.unsqueeze(0).expand(batch_size, seq_len)
        
        # Get embeddings
        token_emb = self.token_embedding(input_ids)  # (B, T, d_model)
        pos_emb = self.position_embedding(positions)  # (B, T, d_model)
        x = token_emb + pos_emb
        
        # Create causal mask (prevents attending to future tokens)
        # True values indicate positions that should be masked
        causal_mask = nn.Transformer.generate_square_subsequent_mask(
            seq_len, device=device
        )
        
        # Pass through transformer decoder
        # Note: TransformerDecoder expects (tgt, memory) but for GPT (decoder-only),
        # we use self-attention only, so we pass x as both tgt and memory
        x = self.transformer(x, x, tgt_mask=causal_mask, memory_mask=causal_mask)
        
        # Final layer norm
        x = self.ln_f(x)
        
        # Project to vocabulary
        logits = self.head(x)  # (B, T, vocab_size)
        
        return logits
```

</details>

<details>
<summary>dataset.py</summary>

```python
from torch.utils.data import Dataset
import torch

class DummyTextDataset(Dataset):
    def __init__(self, num_samples: int = 10000, seq_len: int = 128, vocab_size: int = 50257):
        self.num_samples = num_samples
        self.seq_len = seq_len
        self.vocab_size = vocab_size
    
    def __len__(self):
        return self.num_samples
    
    def __getitem__(self, idx: int):
        # Generate random tokens
        tokens = torch.randint(0, self.vocab_size, (self.seq_len,))
        # For language modeling: input and target (shifted by 1)
        return tokens[:-1], tokens[1:]
```

</details>

For single GPU training, you might write something like this:

```python
# train_single_gpu.py
import torch
import torch.nn as nn
from torch.utils.data import DataLoader
from model import SimpleGPT
from dataset import DummyTextDataset

def train():
    # Set device
    device = torch.device("cuda:0")
    
    # Create model
    model = SimpleGPT(
        vocab_size=50257,
        d_model=512,
        n_heads=8,
        n_layers=6
    ).to(device)
    
    # Optimizer and loss
    optimizer = torch.optim.AdamW(model.parameters(), lr=3e-4)
    criterion = nn.CrossEntropyLoss()
    
    # Create dataset and dataloader
    dataset = DummyTextDataset(num_samples=10000, seq_len=128)
    dataloader = DataLoader(
        dataset,
        batch_size=32,
        shuffle=True,
        num_workers=4,
        pin_memory=True
    )
    
    # Training loop
    num_epochs = 3
    for epoch in range(num_epochs):
        model.train()
        total_loss = 0
        
        for batch_idx, (input_ids, targets) in enumerate(dataloader):
            input_ids = input_ids.to(device)
            targets = targets.to(device)
            
            # Forward pass
            logits = model(input_ids)
            loss = criterion(logits.view(-1, logits.size(-1)), targets.view(-1))
            
            # Backward pass
            optimizer.zero_grad()
            loss.backward()
            torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
            optimizer.step()
            
            total_loss += loss.item()
            
            if batch_idx % 50 == 0:
                print(f"Epoch {epoch}, Batch {batch_idx}, Loss: {loss.item():.4f}")
        
        avg_loss = total_loss / len(dataloader)
        print(f"Epoch {epoch} - Average Loss: {avg_loss:.4f}")
    
    # Save model
    torch.save(model.state_dict(), "checkpoint.pt")
    print("Training complete!")

if __name__ == "__main__":
    train()
```

Now let's see what changes for multi-GPU. I've marked every change with `# CHANGED:` or `# NEW:`.

<details>
<summary>train_multi_gpu.py</summary>

```python
import torch
import torch.nn as nn
import torch.distributed as dist  # NEW: Import distributed
from torch.nn.parallel import DistributedDataParallel as DDP  # NEW: Import DDP
from torch.utils.data import DataLoader
from torch.utils.data.distributed import DistributedSampler  # NEW: Import sampler
import os  # NEW: For environment variables
from model import SimpleGPT

# NEW: Setup function for distributed training
def setup():
    """Initialize the distributed environment."""
    dist.init_process_group(backend="nccl")
    rank = dist.get_rank()
    local_rank = int(os.environ["LOCAL_RANK"])
    world_size = dist.get_world_size()
    torch.cuda.set_device(local_rank)
    return rank, local_rank, world_size

# NEW: Cleanup function
def cleanup():
    """Clean up distributed environment."""
    dist.destroy_process_group()

def train():
    # CHANGED: Setup distributed training
    rank, local_rank, world_size = setup()
    device = torch.device(f"cuda:{local_rank}")  # CHANGED: Use local_rank
    
    # Print info only from rank 0
    if rank == 0:  # NEW: Conditional printing
        print(f"Training on {world_size} GPUs")
    
    # Create model (same as before)
    model = SimpleGPT(
        vocab_size=50257,
        d_model=512,
        n_heads=8,
        n_layers=6
    ).to(device)
    
    # CHANGED: Wrap model with DDP
    model = DDP(model, device_ids=[local_rank])
    
    # Optimizer and loss (same as before)
    optimizer = torch.optim.AdamW(model.parameters(), lr=3e-4)
    criterion = nn.CrossEntropyLoss()
    
    # Create dataset (same as before)
    dataset = DummyTextDataset(num_samples=10000, seq_len=128)
    
    # CHANGED: Use DistributedSampler instead of shuffle=True
    sampler = DistributedSampler(
        dataset,
        num_replicas=world_size,
        rank=rank,
        shuffle=True
    )
    
    # CHANGED: DataLoader uses sampler, no shuffle
    dataloader = DataLoader(
        dataset,
        batch_size=32,
        sampler=sampler,  # CHANGED: Use sampler
        num_workers=4,
        pin_memory=True
    )
    
    # Training loop
    num_epochs = 3
    for epoch in range(num_epochs):
        sampler.set_epoch(epoch)  # NEW: Important for proper shuffling!
        
        model.train()
        total_loss = 0
        
        for batch_idx, (input_ids, targets) in enumerate(dataloader):
            input_ids = input_ids.to(device)
            targets = targets.to(device)
            
            # Forward pass (same as before)
            logits = model(input_ids)
            loss = criterion(logits.view(-1, logits.size(-1)), targets.view(-1))
            
            # Backward pass (same as before, but DDP syncs gradients automatically!)
            optimizer.zero_grad()
            loss.backward()
            torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
            optimizer.step()
            
            total_loss += loss.item()
            
            # CHANGED: Only print from rank 0
            if rank == 0 and batch_idx % 50 == 0:
                print(f"Epoch {epoch}, Batch {batch_idx}, Loss: {loss.item():.4f}")
        
        # NEW: Gather metrics across all GPUs
        avg_loss = total_loss / len(dataloader)
        loss_tensor = torch.tensor([avg_loss]).to(device)
        dist.all_reduce(loss_tensor, op=dist.ReduceOp.AVG)
        
        if rank == 0:  # CHANGED: Only print from rank 0
            print(f"Epoch {epoch} - Average Loss: {loss_tensor.item():.4f}")
    
    # CHANGED: Save only from rank 0 and unwrap DDP
    if rank == 0:
        torch.save(model.module.state_dict(), "checkpoint.pt")
        print("Training complete!")
    
    cleanup()  # NEW: Clean up distributed training

if __name__ == "__main__":
    train()
```

</details>

Let's break down exactly what changed:

<details>
<summary> 1. **Import distributed modules** </summary>

```python
import torch.distributed as dist
from torch.nn.parallel import DistributedDataParallel as DDP
from torch.utils.data.distributed import DistributedSampler
```

These are PyTorch's distributed training primitives:

- `dist`: Provides collective operations (AllReduce, Broadcast, etc.)
- `DDP`: The wrapper that handles automatic gradient synchronization
- `DistributedSampler`: Ensures each GPU processes different data

</details>

<details>
<summary> 2. **Add setup/cleanup functions** </summary>

```python
def setup():
    dist.init_process_group(backend="nccl")
    # Get rank info from environment variables set by torchrun
    ...

def cleanup():
    dist.destroy_process_group()
```

This establishes the communication channels between GPUs:

- `init_process_group()` creates a communication group where all processes can talk to each other
- `backend="nccl"` uses NVIDIA's optimized communication library for GPUs (remember the protocols we discussed?)
- The setup reads `RANK`, `LOCAL_RANK`, `WORLD_SIZE` from environment variables that `torchrun` (we'll see this later) automatically sets
- `cleanup()` properly tears down the communication channels when training finishes

</details>

<details>
<summary> 3. **Set up device** </summary>

```python
# Before:
device = torch.device("cuda:0")

# After:
local_rank = int(os.environ["LOCAL_RANK"])
device = torch.device(f"cuda:{local_rank}")
```

Each process needs to use a different GPU:

- When `torchrun` spawns 4 processes, `LOCAL_RANK` will be 0, 1, 2, 3 for each process
- Process 0 uses GPU 0, process 1 uses GPU 1, etc.
- If all processes tried to use `cuda:0`, they'd all fight over the same GPU while the others sit idle!

</details>

<details>
<summary> 4. **Wrap model with DDP** </summary>

```python
# Before:
model = SimpleGPT(...).to(device)

# After:
model = SimpleGPT(...).to(device)
model = DDP(model, device_ids=[local_rank])  # ← This line does the magic!
```

This is where the distributed magic happens:

- DDP wraps your model and registers hooks on all parameters
- During `backward()`, as each layer computes gradients, DDP immediately starts AllReduce on those gradients
- This overlaps communication with computation (remember the communication vs computation tradeoff?)
- By the time `backward()` finishes, all gradients are already synchronized across GPUs
- Each GPU then has the averaged gradients and updates identically

**Key insight**: You don't manually call AllReduce anywhere - DDP does it automatically during backward pass.

</details>

<details>
<summary> 5. **Dataloader changes** </summary>

```python
# Before:
dataloader = DataLoader(dataset, batch_size=32, shuffle=True)

# After:
sampler = DistributedSampler(dataset, num_replicas=world_size, rank=rank)
dataloader = DataLoader(dataset, batch_size=32, sampler=sampler)
# Note: No shuffle=True, sampler handles it
```

Each GPU must process different data:

- With 4 GPUs and 10,000 samples, `DistributedSampler` ensures:
  - GPU 0 gets samples [0, 4, 8, 12, ...] (2,500 samples)
  - GPU 1 gets samples [1, 5, 9, 13, ...] (2,500 samples)
  - GPU 2 gets samples [2, 6, 10, 14, ...] (2,500 samples)
  - GPU 3 gets samples [3, 7, 11, 15, ...] (2,500 samples)

This is called **data parallelism** - same model, different data per GPU

Without this, all GPUs would process the same data, wasting computation and learning nothing new!
**Why remove shuffle=True?** The sampler handles shuffling in a coordinated way across all GPUs. Using both would cause conflicts.

```python
for epoch in range(num_epochs):
    sampler.set_epoch(epoch)  # ← Important! Ensures different shuffle each epoch
    ...
```

This ensures different data order each epoch:

- `DistributedSampler` uses the epoch number as a random seed
- Without this, every epoch would see data in the same order
- This is bad for training - you want different orderings for better generalization

**What happens if you forget this?**

- Epoch 0: GPU 0 sees [A, E, I, M, ...]
- Epoch 1: GPU 0 sees [A, E, I, M, ...] ← Same order! Bad for training
- Epoch 2: GPU 0 sees [A, E, I, M, ...] ← Still the same!

With `set_epoch()`:

- Epoch 0: GPU 0 sees [A, E, I, M, ...]
- Epoch 1: GPU 0 sees [M, A, I, E, ...] ← Different order!
- Epoch 2: GPU 0 sees [E, M, A, I, ...] ← Different again!

</details>

<details>
<summary> 6. **Conditional printing/saving** </summary>

```python
if rank == 0:  # Only rank 0 prints and saves
    print(...)
    torch.save(model.module.state_dict(), ...)  # Note: model.module, not model
```

- Without it, you'd get 4x duplicate prints (one from each GPU)
- For saving: You'd have 4 processes trying to write to the same file simultaneously, causing corruption or race conditions
- Convention: Rank 0 is the "main" process that handles I/O

Why `model.module`?

- DDP wraps your model, so model is actually a DistributedDataParallel object
- The actual SimpleGPT model is inside at `model.module`
- When saving, you want the underlying model's state dict, not the DDP wrapper's

Without `model.module`: The saved checkpoint would include DDP-specific metadata, making it harder to load for inference or single-GPU training.

</details>

Another big difference is how you run the script.

For single GPU training:

```bash
python train_single_gpu.py
```

**For multi-GPU training:**

```bash
# Use torchrun instead of python
torchrun --nproc_per_node=4 train_multi_gpu.py

# For 2 GPUs:
torchrun --nproc_per_node=2 train_multi_gpu.py

# For 8 GPUs:
torchrun --nproc_per_node=8 train_multi_gpu.py
```

`torchrun` is the built-in tool for multi-GPU training in PyTorch and it automatically:

- Spawns N processes (one per GPU)
- Sets environment variables (`RANK`, `LOCAL_RANK`, `WORLD_SIZE`, `MASTER_ADDR`, `MASTER_PORT`)
- Coordinates the processes
- Handles process failures

### Key Takeaways: The Minimal Diff

To convert single-GPU to multi-GPU, you need just 5 key changes:

- Add: `setup()` and `cleanup()` functions
- Wrap: Model with `DDP(model, device_ids=[local_rank])`
- Replace: `shuffle=True` with `DistributedSampler`
- Add: `sampler.set_epoch(epoch)` in training loop
- Guard: Prints/saves with `if rank == 0:`

The rest of your code stays exactly the same! That's the beauty of PyTorch's distributed training - it's designed to be minimally invasive.

## From PyTorch to Accelerate

After seeing all those changes (setup, cleanup, DDP wrapping, DistributedSampler, conditional saves), you might be thinking: "This is a lot of boilerplate for every project." You're right! That's exactly why Accelerate exists.

**The key benefits:**

- Write once, run anywhere: Same code works on CPU, single GPU, multi-GPU, multi-node, TPU
- Minimal code changes: Even less invasive than manual DDP
- Built-in best practices: Mixed precision, gradient accumulation, checkpoint handling
- Easy configuration: One config file or command-line interface

### Usage of Accelerate

Let's see it in action. Below is the same script as before, but using Accelerate.

<details>
<summary>train_accelerate.py</summary>

```python
import torch
import torch.nn as nn
from torch.utils.data import DataLoader
from accelerate import Accelerator  # That's it for imports!
from model import SimpleGPT
from dataset import DummyTextDataset

def train():
    # Initialize Accelerator - this handles EVERYTHING
    accelerator = Accelerator(
        mixed_precision="bf16",  # Optional: use mixed precision
        gradient_accumulation_steps=2,  # Optional: accumulate gradients
    )
    
    # Create model (no .to(device) needed!)
    model = SimpleGPT(
        vocab_size=50257,
        d_model=512,
        n_heads=8,
        n_layers=6
    )
    
    # Optimizer and loss (same as before)
    optimizer = torch.optim.AdamW(model.parameters(), lr=3e-4)
    criterion = nn.CrossEntropyLoss()
    
    # Create dataset and dataloader (no DistributedSampler needed!)
    dataset = DummyTextDataset(num_samples=10000, seq_len=128)
    dataloader = DataLoader(
        dataset,
        batch_size=32,
        shuffle=True,  # ← Yes, you can use shuffle=True!
        num_workers=4,
        pin_memory=True
    )
    
    # Prepare everything - Accelerator handles all the distributed setup!
    model, optimizer, dataloader = accelerator.prepare(
        model, optimizer, dataloader
    )
    
    # Training loop (looks like single-GPU code!)
    num_epochs = 3
    for epoch in range(num_epochs):
        model.train()
        total_loss = 0
        
        for batch_idx, (input_ids, targets) in enumerate(dataloader):
            # No .to(device) needed - accelerator handles it!
            
            with accelerator.accumulate(model):  # Handles gradient accumulation
                # Forward pass
                logits = model(input_ids)
                loss = criterion(logits.view(-1, logits.size(-1)), targets.view(-1))
                
                # Backward pass - accelerator handles everything!
                accelerator.backward(loss)
                
                # Gradient clipping
                if accelerator.sync_gradients:
                    accelerator.clip_grad_norm_(model.parameters(), 1.0)
                
                optimizer.step()
                optimizer.zero_grad()
            
            total_loss += loss.item()
            
            # Only print from main process (accelerator knows which one)
            if accelerator.is_main_process and batch_idx % 50 == 0:
                print(f"Epoch {epoch}, Batch {batch_idx}, Loss: {loss.item():.4f}")
        
        # Gather metrics (accelerator handles all_reduce)
        avg_loss = total_loss / len(dataloader)
        
        if accelerator.is_main_process:
            print(f"Epoch {epoch} - Average Loss: {avg_loss:.4f}")
    
    # Save model (accelerator handles unwrapping and conditional save)
    accelerator.wait_for_everyone()
    if accelerator.is_main_process:
        unwrapped_model = accelerator.unwrap_model(model)
        torch.save(unwrapped_model.state_dict(), "checkpoint.pt")
        print("Training complete!")

    accelerator.end_training()  # distroy process group

if __name__ == "__main__":
    train()
```

</details>

Let's compare what you had to do manually vs what `Accelerate` handles:

| Task | Manual DDP | Accelerate |
|------|-----------|------------|
| Initialize distributed | `dist.init_process_group()` | `Accelerator()` |
| Get rank/device | `rank = dist.get_rank()` `device = torch.device(f"cuda:{local_rank}")` | Automatic |
| Move model to device | `model.to(device)` | Handled by `prepare()` |
| Wrap with DDP | `model = DDP(model, device_ids=[local_rank])` | Handled by `prepare()` |
| DistributedSampler | Create manually | Handled by `prepare()` |
| Set epoch on sampler | `sampler.set_epoch(epoch)` | Automatic |
| Move data to device | `data.to(device)` for every batch | Automatic |
| Backward pass | `loss.backward()` | `accelerator.backward(loss)` |
| Conditional operations | `if rank == 0:` everywhere | `if accelerator.is_main_process:` |
| Unwrap for saving | `model.module` | `accelerator.unwrap_model(model)` |
| Cleanup | `dist.destroy_process_group()` | `accelerator.end_training()` |

The magic of `prepare()`:

```python
model, optimizer, dataloader = accelerator.prepare(model, optimizer, dataloader)
```

This one line:

- Wraps model with DDP (if multi-GPU)
- Replaces DataLoader's sampler with DistributedSampler (if multi-GPU)
- Sets up mixed precision (if requested)
- Configures gradient accumulation (if requested)
- Handles device placement for everything

Why `accelerator.backward()` instead of `loss.backward()`? Because it handles:

- Mixed precision scaling (if using FP16/BF16)
- Gradient accumulation (only sync on last accumulation step)
- Proper gradient synchronization with DDP

### Running the script

Before running, you should configure `Accelerate`.

You can choose either by building a global config file through:

```bash
accelerate config
```

This asks you questions about your setup:

```plaintext
In which compute environment are you running?
- This machine
- AWS (Amazon SageMaker)

...
```

It creates a config file at `~/.cache/huggingface/accelerate/default_config.yaml`

**Or skip the config and specify at launch**:

```shell
accelerate launch --num_processes=4 --mixed_precision=bf16 train_accelerate.py
```

So here is how we can launch the script:

```shell
# single GPU
accelerate launch train_accelerate.py
# or just
python train_accelerate.py

# Multi-GPU
accelerate launch --num_processes=4 train_accelerate.py
```

### More details of Accelerate

Accelerate offers much more than just simplified DDP. From the example above:

<details>
<summary>Automatic Mixed Precision</summary>

```python
accelerator = Accelerator(mixed_precision="bf16")
# That's it! BF16 training
```

</details>

<details>
<summary>Gradient Accumulation</summary>

```python
accelerator = Accelerator(gradient_accumulation_steps=4)

for batch in dataloader:
    with accelerator.accumulate(model):  # Only syncs every 4 steps
        loss = model(batch)
        accelerator.backward(loss)
        optimizer.step()
        optimizer.zero_grad()
```

Please check the [Gradient Accumulation](https://huggingface.co/docs/accelerate/v1.11.0/en/usage_guides/gradient_accumulation) section for more details.

</details>

<details>
<summary>Easy Checkpointing</summary>

```python
# Save everything (model, optimizer, dataloader state, random states)
accelerator.save_state("checkpoint_dir")

# Load everything back
accelerator.load_state("checkpoint_dir")
```

</details>

<details>
<summary> Logging & Tracker </summary>

```python
# Automatically aggregates metrics across GPUs
accelerator.log({
    "train_loss": loss.item(),
    "learning_rate": optimizer.param_groups[0]['lr']
}, step=global_step)
```

Please check the [Experiment Tracking](https://huggingface.co/docs/accelerate/v1.11.0/en/usage_guides/tracking#implementing-custom-trackers) section for more details.

</details>

Most practitioners use `Accelerate` (or similar abstractions like `PyTorch Lightning`) for production because:

- Less boilerplate = fewer bugs
- Same code runs everywhere
- Built-in optimizations
- Active maintenance by HuggingFace

But understanding manual DDP (like we just did) is crucial because:

- You'll debug issues more effectively
- You'll understand what `Accelerate` is doing
- You can optimize when needed
- You're not mystified by distributed training

I really recommend readers to learn `Accelerate` and become an expert to it since it is a powerful tool for distributed training. You can find more details in the [Accelerate documentation](https://huggingface.co/docs/accelerate/en/index).

## Wrapping Up

You've just gone from zero to hero in multi-GPU training. You now understand how GPUs communicate (PCIe vs NVLink, NCCL, AllReduce), built a GPT model from scratch, and learned two ways to distribute training: torchrun (the standard way), and Accelerate (the easy way). The key insight? Multi-GPU training is just 5 changes: initialize distributed, wrap with DDP, use DistributedSampler, set epoch, and guard prints/saves.

This article covered data parallelism - perfect when your model fits on one GPU. But what about 175B parameter models? That's where model parallelism, tensor parallelism, and pipeline parallelism come in. We'll explore those next. For now, you have the foundation and the intuition to reason about distributed training, not just copy-paste code. That's what matters.
