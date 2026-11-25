## Runs

| run | result |
|---|---|
| attn_l0_zg0 | PASS |
| attn_l0_zg1 | PASS |
| attn_rand_zg0 | PASS |
| attn_rand_zg1 | PASS |
| attn_regression | **FAIL** |
| os_f24_zg0 | PASS |
| os_f24_zg1 | PASS |
| os_f27_zg0 | PASS |
| os_f27_zg1 | PASS |
| os_f29_zg0 | PASS |
| os_f29_zg1 | PASS |
| os_rand_zg0 | PASS |
| os_rand_zg1 | PASS |
| p2_conv | PASS |
| p2_diff_zg0 | PASS |
| p2_diff_zg1 | PASS |
| part1_core_tb | PASS |
| ws_f24_fast_zg0 | PASS |
| ws_f24_fast_zg1 | PASS |
| ws_f24_template | PASS |
| ws_f27_fast_zg0 | PASS |
| ws_f27_fast_zg1 | PASS |
| ws_f27_template | PASS |
| ws_f29_fast_zg0 | PASS |
| ws_f29_fast_zg1 | PASS |
| ws_f29_template | PASS |

## Zero-guard gating (same workload, +zg=0 vs +zg=1; outputs identical in both)

| workload | MACs issued | zero-guarded | a_q toggles | b_q toggles | product toggles | c_q toggles | operand+product toggles | reg write-enable bits |
|---|---|---|---|---|---|---|---|---|
| attn_l0 | 98,304 | 44.9% | 96,944 (-15%) | 29,676 (-15%) | 146,520 (-17%) | 269,515 (+1%) | 273,140 (-16%) | 2,939,352 (-14%) |
| attn_rand | 49,152 | 10.9% | 91,364 (-8%) | 31,996 (-6%) | 150,513 (-8%) | 198,842 (+0%) | 273,873 (-8%) | 1,640,556 (-4%) |
| os_f24 | 294,912 | 84.0% | 33,936 (-62%) | 406,952 (-22%) | 400,702 (+0%) | 119,272 (+0%) | 841,590 (-17%) | 1,976,704 (-72%) |
| os_f27 | 9,216 | 84.0% | 2,080 (-35%) | 15,184 (-12%) | 11,521 (+0%) | 9,792 (+0%) | 28,785 (-10%) | 79,760 (-66%) |
| os_f29 | 589,824 | 75.7% | 225,280 (-53%) | 795,888 (-25%) | 1,058,615 (+0%) | 883,644 (+0%) | 2,079,783 (-20%) | 5,875,792 (-61%) |
| os_rand | 25,600 | 0.0% | 52,704 (+0%) | 50,800 (+0%) | 124,734 (+0%) | 109,968 (+0%) | 228,238 (+0%) | 647,168 (+0%) |
| p2_diff | 933,120 | 34.3% | 1,209,028 (-21%) | 2,279 (+0%) | 2,291,170 (-22%) | 4,973,106 (+0%) | 3,502,477 (-22%) | 20,721,088 (-6%) |
| ws_f24_fast | 663,552 | 88.2% | 228,230 (-27%) | 70,840 (+0%) | 297,920 (-58%) | 1,922,814 (+0%) | 596,990 (-45%) | 35,739,544 (-9%) |
| ws_f27_fast | 20,736 | 87.5% | 9,236 (-18%) | 2,426 (+0%) | 12,098 (-48%) | 61,084 (+0%) | 23,760 (-35%) | 1,104,192 (-9%) |
| ws_f29_fast | 1,327,104 | 79.9% | 643,926 (-37%) | 134,336 (+0%) | 710,202 (-52%) | 3,266,334 (+0%) | 1,488,464 (-43%) | 70,869,256 (-9%) |

## WS vs OS on the VGG layers (measured, zg=0)

| layer | WS (opt.) compute+acc | OS compute | speedup | WS SRAM rd (xmem / pmem) | OS SRAM rd (xmem / wmem) | WS pmem wr | OS pmem wr |
|---|---|---|---|---|---|---|---|
| features[27] | 1,055 | 202 | 5.22x | 396 / 144 | 144 / 144 | 340 | 16 |
| features[24] | 34,256 | 4,678 | 7.32x | 12,672 / 5,104 | 4,608 / 4,608 | 10,880 | 16 |
| features[29] | 67,520 | 12,928 | 5.22x | 25,344 / 9,216 | 9,216 / 9,216 | 21,760 | 1,024 |

## Linear attention core (cycles per head, on chip)

- attn_l0_zg0: 364
- attn_l0_zg1: 364
- attn_rand_zg0: 364
- attn_rand_zg1: 364
