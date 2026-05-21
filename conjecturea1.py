from math import gcd, isqrt

# ============== 1. Sieve ==============

def sieve(n):
    ip = bytearray([1]) * (n + 1)
    ip[0] = ip[1] = 0
    for i in range(2, isqrt(n) + 1):
        if ip[i]:
            ip[i*i::i] = bytearray(len(ip[i*i::i]))
    return ip

X_MAX    = 100_000
Q        = 210
is_prime = sieve(X_MAX + 2)
primes   = [p for p in range(2, X_MAX + 1) if is_prime[p]]

# ============== 2. U_Q ==============

U_Q = [r for r in range(Q) if gcd(r, Q) == 1 and gcd((r + 2) % Q, Q) == 1]

phi_Q = sum(1 for r in range(Q) if gcd(r, Q) == 1)
print("=" * 62)
print(f"  Q     = {Q}  (= 2 × 3 × 5,  squarefree,  6 | Q  ✓)")
print(f"  φ(Q)  = {phi_Q}")
print(f"  U_Q   = {U_Q}  (|U_Q| = {len(U_Q)})")
print(f"  Admissible residue pairs mod Q:")
for r in U_Q:
    print(f"    r = {r:>2}  →  ({r}, {(r+2)%Q})  "
          f"covers e.g. twin primes ≡ {r} mod {Q}")
print()
print("  Exceptional twin-prime pairs excluded from π₂  (prime divides Q):")
for p in [3, 5]:
    if is_prime[p] and is_prime[p+2]:
        print(f"    ({p},{p+2}): {p} mod {Q} = {p%Q},  gcd({p},{Q})={gcd(p,Q)} ≠ 1")
print("=" * 62)
print()

# ============== 3. Helper: B_Q from residue-class counts ==============

def compute_B(pi_Qa):
    return sum(min(pi_Qa[r], pi_Qa[(r + 2) % Q]) for r in U_Q)

# ============== 4. Sweep x = 7 … X_MAX ==============

pi_Qa  = [0] * Q   # π(x; Q, r)
pi2    = 0          # π₂(x)
violations = []

pidx = 0            # index into primes[]
np   = len(primes)

for x in range(2, X_MAX + 1):
    # absorb all primes ≤ x that haven't been absorbed yet
    while pidx < np and primes[pidx] <= x:
        p = primes[pidx]
        if gcd(p, Q) == 1:                          # skip primes dividing Q
            pi_Qa[p % Q] += 1
            # Does p form the UPPER half of a twin-prime pair?
            # i.e., is (p-2, p) a valid pair (both coprime to Q, p-2 prime)?
            if (p >= 9                              # p-2 ≥ 7
                    and is_prime[p - 2]
                    and gcd(p - 2, Q) == 1):
                pi2 += 1
        pidx += 1

    if x < 7:
        continue

    Bx = compute_B(pi_Qa)
    if pi2 > Bx:
        violations.append((x, pi2, Bx))

# ─============== 5. Table ==============

checkpoints = list(range(7, 50)) + [
    50, 100, 200, 500, 1_000, 2_000, 5_000,
    10_000, 20_000, 50_000, 100_000,
]
cp_set = set(checkpoints)

# Re-sweep for display (same logic, clean state)
pi_Qa2 = [0] * Q
pi2_2  = 0
pidx2  = 0

print(f"{'x':>10}  {'π₂(x)':>8}  {'B₃₀(x)':>8}  {'slack':>7}  {'B/π₂':>9}")
print("─" * 52)

for x in range(2, X_MAX + 1):
    while pidx2 < np and primes[pidx2] <= x:
        p = primes[pidx2]
        if gcd(p, Q) == 1:
            pi_Qa2[p % Q] += 1
            if p >= 9 and is_prime[p - 2] and gcd(p - 2, Q) == 1:
                pi2_2 += 1
        pidx2 += 1

    if x in cp_set:
        Bx = compute_B(pi_Qa2)
        ratio = f"{Bx/pi2_2:.3f}" if pi2_2 > 0 else "      —"
        print(f"{x:>10,}  {pi2_2:>8}  {Bx:>8}  {Bx-pi2_2:>7}  {ratio:>9}")

print("─" * 52)
print()

# ============== 6. Verdict ==============

if violations:
    print(f"❌  VIOLATED at {len(violations)} point(s) in [7, {X_MAX:,}].")
    for x, p2, Bx in violations[:15]:
        print(f"   x={x:,}: π₂={p2}, B₃₀={Bx}, deficit={p2-Bx}")
else:
    print(f"✅  π₂(x) ≤ B₃₀(x) holds for ALL x in [7, {X_MAX:,}].  "
          "No violations found.")

# ============== 7. Residue spot-check ==============

print()
print("Residue classes of twin prime pairs (p, p+2), 7 ≤ p ≤ 500:")
print(f"  {'(p, p+2)':>12}  {'p mod 30':>9}  {'(p+2) mod 30':>13}  ∈ U_Q?")
for p in range(7, 501):
    if is_prime[p] and is_prime[p + 2] and gcd(p, Q) == 1:
        r = p % Q
        ok = "✓" if r in U_Q else "✗  ← UNEXPECTED"
        print(f"  ({p:>3}, {p+2:>3})        {r:>9}  {(r+2)%Q:>13}     {ok}")
