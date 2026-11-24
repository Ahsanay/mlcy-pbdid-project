"""
Step 1: build the common feature set from UNSW-NB15 (raw) and Bot-IoT (5%).

The paper (Zeeshan et al., 2022, Table 4) trains on 26 features that exist
in Bot-IoT and can be derived for UNSW-NB15. Bot-IoT already has all 26.
UNSW-NB15 has 14 of them directly; the other 12 are the Bot-IoT style
"per 100 connections" statistics, which we compute here.

Output (data/processed/):
    unsw_26.parquet    all 2.54M UNSW-NB15 records, 26 features + label
    botiot_26.parquet  all 3.67M Bot-IoT 5% records, 26 features + label

Run:  python src/prepare_data.py
"""
import glob
import os

import numpy as np
import pandas as pd

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
RAW_UNSW = os.path.join(ROOT, "data", "unsw_nb15_raw")
RAW_BOT = os.path.join(ROOT, "data", "bot_iot")
OUT = os.path.join(ROOT, "data", "processed")

# Table 4 of the paper. First 9 are the "flow" cluster, the rest "TCP".
FEATURES = [
    "stime", "ltime", "saddr", "daddr", "dur",
    "AR_P_Proto_P_SrcIP", "AR_P_Proto_P_DstIP",
    "N_IN_Conn_P_SrcIP", "N_IN_Conn_P_DstIP",
    "sport", "dport", "pkts", "bytes", "proto_number",
    "spkts", "dpkts", "sbytes", "dbytes",
    "TnBPSrcIP", "TnBPDstIP", "TnP_PSrcIP", "TnP_PDstIP",
    "TnP_PerProto", "TnP_Per_Dport",
    "AR_P_Proto_P_Sport", "AR_P_Proto_P_Dport",
]

# Bot-IoT numbering for the protocols it contains. Protocols that only
# appear in UNSW-NB15 get the next free integers.
PROTO_NUMBER = {"tcp": 1, "arp": 2, "udp": 3, "icmp": 4, "ipv6-icmp": 5}

WINDOW = 100  # Bot-IoT statistics are computed over 100 connections


def port_to_int(col):
    """Ports are mostly integers, a few are hex strings, '-' means none."""
    s = col.astype(str).str.strip()
    out = pd.to_numeric(s, errors="coerce")
    is_hex = s.str.startswith("0x")
    out[is_hex] = s[is_hex].apply(lambda v: int(v, 16))
    return out.fillna(-1).astype("int64")  # -1 as in the Bot-IoT 5% files


def load_unsw():
    names = pd.read_csv(os.path.join(RAW_UNSW, "NUSW-NB15_features.csv"),
                        encoding="latin-1")["Name"].str.strip().tolist()
    files = sorted(glob.glob(os.path.join(RAW_UNSW, "UNSW-NB15_?.csv")))
    df = pd.concat([pd.read_csv(f, header=None, names=names, low_memory=False)
                    for f in files], ignore_index=True)

    out = pd.DataFrame()
    out["stime"] = df["Stime"].astype("float64")
    out["ltime"] = df["Ltime"].astype("float64")
    out["saddr"] = df["srcip"].astype(str)
    out["daddr"] = df["dstip"].astype(str)
    out["dur"] = df["dur"].astype("float64")
    out["sport"] = port_to_int(df["sport"])
    out["dport"] = port_to_int(df["dsport"])
    out["spkts"] = df["Spkts"].astype("int64")
    out["dpkts"] = df["Dpkts"].astype("int64")
    out["sbytes"] = df["sbytes"].astype("int64")
    out["dbytes"] = df["dbytes"].astype("int64")
    out["pkts"] = out["spkts"] + out["dpkts"]
    out["bytes"] = out["sbytes"] + out["dbytes"]
    out["proto"] = df["proto"].astype(str).str.strip().str.lower()

    cat = df["attack_cat"].fillna("Normal").astype(str).str.strip()
    cat = cat.replace({"": "Normal", "Backdoors": "Backdoor"})
    out["category"] = cat

    # time order is needed for the 100-connection windows
    out = out.sort_values("stime", kind="stable").reset_index(drop=True)
    return out


def add_window_features(df):
    """Bot-IoT style statistics over consecutive windows of 100 records.

    Definitions follow the Bot-IoT paper (Koroniotis et al., 2019):
      TnBPSrcIP / TnBPDstIP       total bytes per source / destination IP
      TnP_PSrcIP / TnP_PDstIP     total packets per source / destination IP
      TnP_PerProto                total packets per protocol
      TnP_Per_Dport               total packets per destination port
      AR_P_Proto_P_*              average rate (packets / duration) per
                                  protocol per src IP, dst IP, sport, dport
      N_IN_Conn_P_SrcIP / DstIP   number of connections per src / dst IP
    """
    w = np.arange(len(df)) // WINDOW
    df = df.assign(_w=w)

    def total(keys, col):
        return df.groupby(["_w"] + keys)[col].transform("sum")

    df["TnBPSrcIP"] = total(["saddr"], "bytes")
    df["TnBPDstIP"] = total(["daddr"], "bytes")
    df["TnP_PSrcIP"] = total(["saddr"], "pkts")
    df["TnP_PDstIP"] = total(["daddr"], "pkts")
    df["TnP_PerProto"] = total(["proto"], "pkts")
    df["TnP_Per_Dport"] = total(["dport"], "pkts")

    for name, key in [("AR_P_Proto_P_SrcIP", "saddr"),
                      ("AR_P_Proto_P_DstIP", "daddr"),
                      ("AR_P_Proto_P_Sport", "sport"),
                      ("AR_P_Proto_P_Dport", "dport")]:
        p = total(["proto", key], "pkts")
        d = total(["proto", key], "dur")
        df[name] = np.where(d > 0, p / d.where(d > 0, 1.0), 0.0)

    df["N_IN_Conn_P_SrcIP"] = df.groupby(["_w", "saddr"])["pkts"].transform("size")
    df["N_IN_Conn_P_DstIP"] = df.groupby(["_w", "daddr"])["pkts"].transform("size")
    return df.drop(columns="_w")


def load_botiot():
    files = sorted(glob.glob(os.path.join(RAW_BOT, "UNSW_2018_IoT_Botnet_Full5pc_?.csv")))
    df = pd.concat([pd.read_csv(f, low_memory=False) for f in files],
                   ignore_index=True)
    df["sport"] = port_to_int(df["sport"])
    df["dport"] = port_to_int(df["dport"])
    df["saddr"] = df["saddr"].astype(str)
    df["daddr"] = df["daddr"].astype(str)
    df["proto"] = df["proto"].astype(str).str.lower()
    df = df.sort_values("pkSeqID", kind="stable").reset_index(drop=True)
    return df


def main():
    os.makedirs(OUT, exist_ok=True)

    print("loading UNSW-NB15 raw ...")
    unsw = load_unsw()
    print("  records:", len(unsw))
    print("computing 100-connection statistics for UNSW-NB15 ...")
    unsw = add_window_features(unsw)

    print("loading Bot-IoT 5% ...")
    bot = load_botiot()
    print("  records:", len(bot))

    # protocol -> integer, shared by both datasets
    proto_map = dict(PROTO_NUMBER)
    for p in sorted(set(unsw["proto"]) | set(bot["proto"])):
        if p not in proto_map:
            proto_map[p] = len(proto_map) + 1
    unsw["proto_number"] = unsw["proto"].map(proto_map).astype("int64")
    bot["proto_number"] = bot["proto"].map(proto_map).astype("int64")

    # IP address -> random unique integer, one table for both datasets
    # (paper, section III-D-1)
    ips = sorted(set(unsw["saddr"]) | set(unsw["daddr"])
                 | set(bot["saddr"]) | set(bot["daddr"]))
    rng = np.random.default_rng(42)
    ip_map = dict(zip(ips, rng.permutation(len(ips)) + 1))
    print("  unique IP addresses in both datasets:", len(ips))
    for d in (unsw, bot):
        d["saddr"] = d["saddr"].map(ip_map).astype("int64")
        d["daddr"] = d["daddr"].map(ip_map).astype("int64")

    unsw["source"] = "UNSW-NB15"
    bot["source"] = "Bot-IoT"
    keep = FEATURES + ["category", "source"]
    unsw[keep].to_parquet(os.path.join(OUT, "unsw_26.parquet"), index=False)
    bot[keep].to_parquet(os.path.join(OUT, "botiot_26.parquet"), index=False)

    print("\nUNSW-NB15 classes\n", unsw["category"].value_counts().to_string())
    print("\nBot-IoT 5% classes\n", bot["category"].value_counts().to_string())
    print("\nsaved to", OUT)


if __name__ == "__main__":
    main()
