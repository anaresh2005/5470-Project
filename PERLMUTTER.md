# Running on Perlmutter

Everyone uses their own NERSC login and clone of this repo. Questions: ask Nathnael.

## 1. Check your access

```
ssh <your-nersc-username>@perlmutter.nersc.gov
iris
```

In the `m4999_g` row, `Allocated(user)` should be higher than `Charged(user)`.
Tell Nathnael your NERSC username and whether this worked.

## 2. Test the setup

```
git clone https://github.com/anaresh2005/5470-Project.git ~/5470-Project
cd ~/5470-Project
source scripts/env.sh && make
sbatch scripts/run_1node.sbatch
```

When the job is gone from `squeue -u $USER`, run `tail logs/allreduce4-*.out`. The last line should start with `PASS`.

This runs the repo's StragglAR, not your code, so `PASS` only means your setup works.

## 3. Run your own program

From the repo root:

```
source scripts/env.sh
mkdir -p build
nvcc -std=c++17 -O3 -diag-suppress=177 -I$NCCL_HOME/include -o build/my_program my_program.cu -L$NCCL_HOME/lib -lnccl
sbatch --export=ALL,BIN=build/my_program scripts/run_1node.sbatch 12582912 stragglar 10 -1
```

The last four values are buffer bytes (keep divisible by 48), algorithm, iterations, and delay in ms (`-1` skips the pre-work phase).

To get `PASS`, your program must:
- accept those four arguments
- print `Rank 0, done` through `Rank 3, done`
- print a line like `Rank 0, idx 5, val 2.000` for each wrong value

`PASS` doesn't compare results against `ncclAllReduce`.

## Good to know

- Jobs are limited to one node and 10 minutes.
- Git ignores `logs/`, so keep a copy of any log behind a poster or report number.