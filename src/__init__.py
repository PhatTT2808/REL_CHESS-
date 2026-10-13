"""Neural-Symbolic Xiangqi Reinforcement Learning.

Package gốc chứa toàn bộ mã nguồn của đồ án:
- ``src.env``: môi trường cờ tướng (Gym-like wrapper + graph builder).
- ``src.models``: GNN encoder, motif attention, policy network.
- ``src.training``: reward, PPO, self-play.
- ``src.evaluation``: Elo, đấu với engine UCI.
- ``src.utils``: cấu hình, seed, checkpoint.
"""

__version__ = "0.1.0"
