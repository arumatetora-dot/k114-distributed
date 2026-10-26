import os
import sys
import time
import math
import random
import argparse
import numpy as np
import gmpy2
from gmpy2 import mpz, isqrt as gmp_isqrt
import numba
from numba import njit, prange

# =====================================================================
# 1. 前置フィルター用 O(1) テーブル群
# =====================================================================
PRIMES_CM = np.array([7, 13, 19, 31, 37, 43, 61, 67, 73, 79, 97, 103], dtype=np.int64)
MAX_Q = 104
AQ_TABLE = np.zeros((len(PRIMES_CM), MAX_Q), dtype=np.int64)
S_MN_WEIGHT = np.zeros((len(PRIMES_CM), MAX_Q), dtype=np.int64)

for idx, q in enumerate(PRIMES_CM):
    qr_count = np.zeros(q, dtype=np.int64)
    for y in range(q): qr_count[(y * y) % q] += 1
    for d_mod in range(1, q):
        pts = 1
        for x in range(q): pts += qr_count[(x * x * x + d_mod) % q]
        AQ_TABLE[idx, d_mod] = q + 1 - pts
        
    log_factor = math.log(float(q)) / float(q)
    for s_mod in range(q):
        s3 = pow(s_mod, 3, int(q))
        d_mod = (432 * s3 * (456 - s3)) % int(q)
        if d_mod != 0:
            aq = int(AQ_TABLE[idx, d_mod])
            S_MN_WEIGHT[idx, s_mod] = int(round((-aq * log_factor) * 10000.0))

PRIMES_QR = np.array([5, 7, 11, 13, 17, 19, 23, 29, 31, 37, 41, 43, 47, 53, 59, 61], dtype=np.int64)
MASKS_QR = np.zeros(16, dtype=np.uint64)
INV_TABLE = np.zeros((16, 62), dtype=np.int64)
CUBES_TABLE = np.zeros((16, 62), dtype=np.int64)

for k_idx, p in enumerate(PRIMES_QR):
    m_mask = np.uint64(0)
    for i in range(p): m_mask |= np.uint64(1) << np.uint64((i * i) % p)
    MASKS_QR[k_idx] = m_mask
    for x in range(p):
        CUBES_TABLE[k_idx, x] = (x * x * x) % p
        if x != 0: INV_TABLE[k_idx, x] = pow(x, -1, int(p))

# =====================================================================
# 2. 超並列 JIT コアカーネル
# =====================================================================
@njit
def mod_inverse_safe(a, m):
    m0, y, x = m, 0, 1
    if m <= 1: return 0
    a = a % m
    while a > 1 and m > 0:
        q = a // m
        a, m = m, a % m
        x, y = y, x - q * y
    if a != 1: return -1
    if x < 0: x += m0
    return x

@njit(parallel=True)
def cpu_sieve_four_ideas_chunk(A_vals, R_As, A_mod9s, B_vals, R_Bs, B_mod9s, m_max, primes_qr, masks_qr, inv_table, cubes_table, primes_cm, s_mn_weight, mn_cutoff_scaled):
    num_A, num_B = len(A_vals), len(B_vals)
    num_threads = numba.get_num_threads()
    local_cap = 5000
    local_s = np.zeros((num_threads, local_cap), dtype=np.int64)
    local_z = np.zeros((num_threads, local_cap), dtype=np.int64)
    local_counts = np.zeros(num_threads, dtype=np.int32)

    for i in prange(num_A):
        tid = numba.get_thread_id()
        a_val = A_vals[i]
        r_a = R_As[i]
        a_mod9 = A_mod9s[i]

        for j in range(num_B):
            b_val = B_vals[j]
            b_mod9 = B_mod9s[j]

            rem9 = (a_mod9 * b_mod9) % 9
            if rem9 == 0 or rem9 == 3 or rem9 == 6: continue

            d = np.int64(a_val) * np.int64(b_val)
            s = d if (rem9 == 1 or rem9 == 4 or rem9 == 7) else -d

            mn_sum = np.int64(0)
            for idx_q in range(6):
                q_cm = primes_cm[idx_q]
                s_mod_q = s % q_cm
                if s_mod_q < 0: s_mod_q += q_cm
                mn_sum += s_mn_weight[idx_q, s_mod_q]
            if mn_sum < mn_cutoff_scaled: continue

            inv_b = mod_inverse_safe(b_val, a_val)
            if inv_b < 0: continue

            diff = (r_a - R_Bs[j]) % a_val
            if diff < 0: diff += a_val
            r_ab = R_Bs[j] + ((diff * inv_b) % a_val) * b_val
            z_base = r_ab if s < 0 else (r_ab - d)
            step_d = d if s < 0 else -d

            for m_idx in range(1, m_max + 1):
                z = z_base + m_idx * step_d
                if ((z + s) % 9 + 9) % 9 != 0: continue

                passed_qr = True
                for k_idx in range(16):
                    q = primes_qr[k_idx]
                    s_q = s % q
                    if s_q < 0: s_q += q
                    z_q = z % q
                    if z_q < 0: z_q += q

                    s3_val = (3 * s_q) % q
                    z3 = cubes_table[k_idx, z_q]
                    s_cubed = cubes_table[k_idx, s_q]
                    val = (456 - 4 * z3 - s_cubed) % q
                    if val < 0: val += q

                    if s3_val == 0:
                        if val != 0: passed_qr = False; break
                    else:
                        inv = inv_table[k_idx, s3_val]
                        u_sq = (val * inv) % q
                        if ((masks_qr[k_idx] >> np.uint64(u_sq)) & np.uint64(1)) == 0:
                            passed_qr = False; break

                if passed_qr:
                    c = local_counts[tid]
                    if c < local_cap:
                        local_s[tid, c] = s
                        local_z[tid, c] = z
                        local_counts[tid] = c + 1
    return local_s, local_z, local_counts

# =====================================================================
# 3. 動的ブロック生成 ＆ GMP監査
# =====================================================================
def get_primes_mod2(n):
    primes, p = [], 2
    while len(primes) < n:
        if p % 3 == 2 and all(p % d != 0 for d in range(2, int(p**0.5) + 1)):
            primes.append(p)
        p += 1
    return primes

def build_random_blocks(prime_group, num_samples, roots_mod2, max_val):
    vals, rs, mod9s = [], [], []
    seen = set()
    p_list = list(prime_group)
    n_p = len(p_list)
    
    while len(vals) < num_samples:
        k = random.choice([2, 3, 4])
        selected_idx = random.sample(range(n_p), k)
        M = 1
        valid = True
        for idx in selected_idx:
            M *= p_list[idx]
            if M > max_val:
                valid = False; break
        if not valid or M in seen: continue
            
        r_val = 0
        for idx in selected_idx:
            p = p_list[idx]
            M_p = M // p
            r_val = (r_val + roots_mod2[p] * M_p * pow(M_p, -1, p)) % M
            
        seen.add(M)
        vals.append(M)
        rs.append(r_val)
        mod9s.append(M % 9)
        
    return np.array(vals, dtype=np.int64), np.array(rs, dtype=np.int64), np.array(mod9s, dtype=np.int64)

def verify_survivors(s_list, z_list, K_TARGET=114):
    sols = []
    for s_val, z_val in zip(s_list, z_list):
        s_mp, z_mp = mpz(int(s_val)), mpz(int(z_val))
        lhs = 4 * (K_TARGET - z_mp**3)
        if lhs % s_mp != 0: continue
        det_M = lhs // s_mp
        delta = det_M - s_mp * s_mp
        if delta > 0 and delta % 3 == 0:
            u_sq = delta // 3
            if gmpy2.is_square(u_sq):
                u_root = gmp_isqrt(u_sq)
                if (s_mp + u_root) % 2 == 0:
                    sols.append((int((s_mp + u_root)//2), int((s_mp - u_root)//2), int(z_mp), int(s_mp)))
    return sols

# =====================================================================
# 4. メイン：分散Worker (担当の100億ペアを処理)
# =====================================================================
if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--worker_id", type=int, required=True, help="WorkerのID")
    parser.add_argument("--total_workers", type=int, default=10, help="全Worker数")
    parser.add_argument("--time_limit", type=int, default=600, help="テスト時間/秒")
    args = parser.parse_args()

    wid = args.worker_id
    tot = args.total_workers

    print(f"[{time.strftime('%H:%M:%S')}] 🚀 Worker {wid}/{tot-1} 起動：10分間 100億ペア分担テスト開始")

    # 全ノードで共通の宇宙を展開するためのシード固定
    random.seed(114)

    primes = get_primes_mod2(500)
    roots = {p: pow(114, (2 * p - 1) // 3, p) for p in primes}
    
    group_B = [p for i, p in enumerate(primes) if i % 5 == 0]
    group_A = [p for i, p in enumerate(primes) if i % 5 != 0]

    MAX_A = 3_000_000_000
    MAX_B = 3_000_000_000
    MN_CUTOFF = np.int64(0)
    
    # 宇宙の総サイズ：A=100万, B=10万 (1000億ペア)
    TOTAL_A = 1000000
    TOTAL_B = 100000
    
    A_vals_full, R_As_full, A_mod9s_full = build_random_blocks(group_A, TOTAL_A, roots, MAX_A)
    B_vals, R_Bs, B_mod9s = build_random_blocks(group_B, TOTAL_B, roots, MAX_B)
    
    # Worker ごとに自分の担当領域 (100億ペア) だけを切り出す
    chunk_size = TOTAL_A // tot
    start_idx = wid * chunk_size
    end_idx = start_idx + chunk_size if wid < tot - 1 else TOTAL_A

    A_vals = A_vals_full[start_idx:end_idx]
    R_As = R_As_full[start_idx:end_idx]
    A_mod9s = A_mod9s_full[start_idx:end_idx]
    
    my_A_len = len(A_vals)
    my_space = my_A_len * len(B_vals)
    print(f"[{time.strftime('%H:%M:%S')}] Worker {wid}: 担当インデックス {start_idx}〜{end_idx-1} | 担当空間: {my_space:,} ペア")

    _ = cpu_sieve_four_ideas_chunk(A_vals[:2], R_As[:2], A_mod9s[:2], B_vals[:2], R_Bs[:2], B_mod9s[:2], 1, PRIMES_QR, MASKS_QR, INV_TABLE, CUBES_TABLE, PRIMES_CM, S_MN_WEIGHT, MN_CUTOFF)

    start_time = time.time()
    last_report = start_time
    total_evaluated = 0
    total_audits = 0
    CHUNK_A_RUN = 500

    try:
        for i in range(0, my_A_len, CHUNK_A_RUN):
            if time.time() - start_time >= args.time_limit:
                print(f"\n[Worker {wid}] 設定タイムリミット（{args.time_limit}秒）到達。")
                break
                
            e = min(i + CHUNK_A_RUN, my_A_len)
            
            loc_s, loc_z, counts = cpu_sieve_four_ideas_chunk(
                A_vals[i:e], R_As[i:e], A_mod9s[i:e], 
                B_vals, R_Bs, B_mod9s, 10, PRIMES_QR, MASKS_QR, INV_TABLE, CUBES_TABLE, PRIMES_CM, S_MN_WEIGHT, MN_CUTOFF
            )
            
            pairs = (e - i) * len(B_vals)
            total_evaluated += pairs
            
            for tid in range(len(counts)):
                c = counts[tid]
                if c > 0:
                    total_audits += int(c)
                    sols = verify_survivors(loc_s[tid, :c], loc_z[tid, :c])
                    for x, y, z, s in sols:
                        msg = f"\n🚨 [Worker {wid} K=114 解発見!] x={x}, y={y}, z={z}"
                        print(msg)
                        with open(f"K114_WORKER_{wid}_SOLUTION.txt", "w") as f: f.write(msg)

            if time.time() - last_report >= 60.0:
                elapsed = time.time() - start_time
                speed = total_evaluated / elapsed
                pct = (e / my_A_len) * 100
                print(f"[Worker {wid}] {elapsed/60:.1f}分経過 | 進捗: {pct:5.1f}% | 速度: {speed/1e8:.2f} 億ペア/秒 | GMP: {total_audits}")
                last_report = time.time()
                
    except KeyboardInterrupt:
        pass

    elapsed = time.time() - start_time
    print(f"\n🏁 [Worker {wid} 完了] 稼働: {elapsed:.1f}秒 | 評価数: {total_evaluated:,} ペア | 速度: {total_evaluated/elapsed:,.0f} ペア/秒")
