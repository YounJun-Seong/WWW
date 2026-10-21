"""RX 32B replication: rx_run.py unchanged, with the frozen round-0 taken from the Qwen2.5-32B FRAMES final run
(runs/backbones/qwen2.5-32b, W3b), so R1/R2 interventions act on the 32B agents' own initial answers. Model and server
come from the environment (WQ_MODEL = Qwen/Qwen2.5-32B-Instruct, TP=2 as in W3b). FRAMES only (32B ran FRAMES only)."""
import rx_run

rx_run.FROZEN = dict(frames="runs/backbones/qwen2.5-32b/frames_final_gen.jsonl")
if __name__ == "__main__":
    rx_run.main()
