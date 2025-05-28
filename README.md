# CARV: Constraint-Aware Refinement for Verification

## About
Constraint-Aware Refinement for Verification (CARV) is an approach to efficiently reduce conservativeness in reachable set over-approximations (RSOAs) for verifying the safety of neural feedback loops (NFLs), i.e., closed-loop systems with neural network controllers. The code provided here includes the core algorithm, example configurations, and scripts to reproduce the results presented in our paper:

- Nicholas Rober, Jonathan P. How ["Constraint-Aware Refinement for Safety Verification of Neural Feedback Loops"](https://arxiv.org/abs/2410.00145)

<figure style="margin: 0;">
    <img src="nfl_robustness_training/src/plots/forward/LCSS24/Quadrotor_NL/Quadrotor_NL_CARV15.gif" alt="Quadrotor Animation" style="display: block; margin: 0 auto; object-fit: cover; width: 100%; height: auto; clip-path: inset(100px 150px 80px 200px); overflow: hiddenz;">
</figure>

## Installation

To install the necessary dependencies, run the following command:

```bash
pip install -r requirements.txt
```

## Usage

To run the CARV algorithm, use the following command:

```bash
python run_carv.py --config config.yaml
```

## Examples

You can find example configurations and usage in the `examples` directory.

## License

This project is licensed under the MIT License. See the [LICENSE](LICENSE) file for details.

## Acknowledgements

We would like to thank all contributors and the open-source community for their support.