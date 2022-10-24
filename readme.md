# Scope

This project contains code (written in Python3) for simulating two mechanisms which modify the operation of IEEE 802.11ax UORA OBO countdown:

- Efficient OBO (E-OBO) -- proposed in K. Kosek-Szott and K. Domino, “[An Efficient Backoff Procedure for IEEE 802.11 ax Uplink OFDMA-Based Random Access](https://ieeexplore.ieee.org/abstract/document/9669929),” IEEE Access, vol. 10, pp. 8855–8863, 2022.
- Reinforcement Learning OBO (RL-OBO) -- proposed in K. Kosek-Szott, S. Szott and F. Dressler, "Improving IEEE 802.11ax UORA Performance: Comparison of Reinforcement Learning and Heuristic
  Approaches" (under review).

# E-OBO

The following files are necessary to simulate E-OBO operation:

- `Times.py` -- methods which calculate IEEE 802.11 frame transmission durations.
- `UORA.py` -- a simulator of the IEEE 802.11ax UORA frame exchange, extended to support the E-OBO mechanism.
- `EOBO.py` -- allows running static or dynamic simulations.

The simulation can be run as follows:

```python
./python EOBO.py
```

Within the code, the `dynamic` boolean defines which scenario is run (static or dynamic). 

# RL-OBO

The following files are necessary to simulate RL-OBO operation:

- `Times.py` -- methods which calculate 802.11 frame transmission durations.
- `UORA_RL.py` -- a simulator of the IEEE 802.11ax UORA frame exchange, extended to support RL methods.
- `RL-OBO.py` -- implements the RL-OBO mechanism and, if executed, runs the simulation (training, then testing).

The simulation can be run as follows:

```python
./python RL-OBO.py
```

Additionally, to perform a reward parameter sweep (Appendix A in the RL-OBO paper), execute `RL-OBO_reward_params` (which imports `UORA_RL_reward_params`) as follows:

```python
./ipython RL-OBO_reward_params.py
```

