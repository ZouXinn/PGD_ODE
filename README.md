## This is the code that reproduces our experiments

### Step 1. Simulate
Here we need to run the simulations to get simulated results.

Just execute the run_epss.sh file:

``` shell
./run_epss.sh
```

### Step 2. Draw pictures

**Draw Figure 1.**

``` shell
python draw_norm_compare.py
```

**Draw Figure 6 (Left).**

``` shell
python fig6_left.py
```

**Draw Figure 6 (Right).**
``` shell
python fig6_right.py
```

**Draw Figure 7(a).**

``` shell
python draw_path.py
```

**Draw Figure 7(b).**

``` shell
python draw_norm_compare.py --x0 1
```

**Draw Figure 8(a).**

``` shell
python ode_figure8a.py
```

**Draw Figure 8(b).**

``` shell
python ode_figure8b.py
```

**Draw Figure 9.**

``` shell
python draw_norm_epss.py --x0 1
```


**Draw Figure 10.**

``` shell
python fig10.py
```

**Draw Figure 11.**

``` shell
python draw_norm_epss.py
```

**Draw Figure 12.**

``` shell
python fig12.py
```