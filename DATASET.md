# About the Dataset

This project uses two real, public datasets of actual Bitcoin transactions. Neither is made up or simulated — every transaction in them really happened on the Bitcoin blockchain.

## 1. The Elliptic dataset (the main one)

**What it is:** A map of 203,769 real Bitcoin transactions, drawn as a graph — each transaction is a dot ("node"), and a line ("edge") connects two transactions if money flowed from one to the other. There are 234,355 of these connecting lines.

**Who made it, and why:** A company called **Elliptic** — they specialize in tracking illegal activity on the blockchain for banks, governments, and crypto exchanges. They built this dataset together with **MIT and IBM's joint AI research lab (the MIT-IBM Watson AI Lab)**. They released it publicly so researchers could build and test fraud-detection methods. It was first described in a research paper in 2019: *"Anti-Money Laundering in Bitcoin: Experimenting with Graph Convolutional Networks for Financial Forensics"* by Mark Weber and colleagues, presented at KDD (a major data-science research conference).

**How the labels were made:** Elliptic manually identified real transactions belonging to two kinds of activities:
- **Illicit** (about 4,545 transactions, ~2%): things like scams, malware payments, terrorist financing, ransomware, Ponzi schemes.
- **Licit** (about 42,019 transactions, ~21%): things like legitimate exchanges, wallet providers, mining pools, legal services.
- **Unknown** (the remaining ~77%): Elliptic didn't have enough information to label these one way or the other — this is normal in real-world data, not a mistake.

**Why most of the transaction details are hidden:** Each transaction comes with 165 numbers describing it (things like transaction size, number of inputs/outputs, and statistics about its neighboring transactions). Elliptic scrambled/transformed these numbers on purpose before releasing the data, so nobody could reverse-engineer their exact fraud-detection methods or expose real people's financial details. This means we can't say "feature #12 is the transaction amount" — we only know the numbers exist and can be fed into a model, not exactly what each one means.

**Time factor:** Transactions are grouped into 49 time steps (think of it like 49 snapshots of the blockchain taken over time). This matters: to test a model fairly, you have to train it on the *earlier* time steps and test it on *later* ones — otherwise the model can "cheat" by learning from the future. The train/val/test split throughout this project is by time step, never random.

## 2. Elliptic++ (an extension, used for the "does this work at a bigger scale" experiment)

**What it is:** A bigger, follow-up version of the same idea, but built around **Bitcoin wallet addresses** instead of individual transactions — 822,942 wallets, connected by 2.87 million relationships (who sent money to whom).

**Who made it:** Researchers at Georgia Tech (Youssef Elmougy and Ling Liu), published at KDD 2023, building on top of Elliptic's original release with the same research spirit — public, free, for research use.

**Why this project uses it:** To test whether the fraud-detection approach still works when the graph is roughly 4x bigger.

## How the data enters this project (technically)

- Base Elliptic is loaded through `torch_geometric.datasets.EllipticBitcoinDataset` — a ready-made loader built into a popular open-source graph machine-learning library, which fetches the data straight from Elliptic's own file servers. No login, no scraping, fully public.
- Elliptic++ isn't available through that shortcut, so its files were downloaded directly from the researchers' own public GitHub-linked storage.
- Every single count (203,769 nodes, 234,355 edges, 49 time steps, etc.) is double-checked in code against the numbers Elliptic and the Elliptic++ authors published, before any model is trained on it — see `data/validate_phase1.py` and `data/prepare_elliptic_pp.py`.

## What this dataset is *not*

- It is **not** a list of real people's names, addresses, or personal information — Bitcoin wallets are just strings of letters/numbers, not identities.
- It is **not** something this project collected — it's a well-known, widely-cited academic dataset used by many other researchers and companies.
- It is **not** simulated or fake — every transaction in it happened for real on the Bitcoin network.
