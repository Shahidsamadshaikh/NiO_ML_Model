# NiO_ML_Model
[![DOI](https://zenodo.org/badge/1279919903.svg)](https://doi.org/10.5281/zenodo.20963050)
## Classical and Quantum Kernel Model for Band gap prediction of Bulk NiO(Nickel Oxide)

### Overview 
##### This Repository cantained the Trained Classical and Quantum Kernel Model of band gap for NiO(Nickel oxide) 

**Key Results:**
| Model | R² (LOOCV) | RMSE (eV) | MAE (eV) |
|-------|------------|-----------|----------|
| Classical RBF Kernel | 0.9834 | 0.0336 | 0.0336 |
| Quantum Kernel (2-qubit) | 0.9775 | 0.0391 | 0.0391 |

**Paper:** *"Comparative Investigation of Band gaps in Strongly Correlated Quantum Material NiO(Nickel oxide) using Classical and Quantum kernels from DFT+U"* – Submitted to *Nature Computational Materials*.


## Structure of Repository 
#### Repository contained 3 important folders 1) models 2)scripts 3) Result
### 1.models/
##### models folder consist of both classical and Quantum kernel model in .pkl format 
### 2,scripts/ 
##### scripts folder consist of both classical and Quantum model python code 
### 3.Result/
##### Result folder consist of plot of prediction value of band gap by both classical and Quantum kernel using unknown value of U from 5 eV to 6.8 eV 

## Citation
If you use this code, please cite:
> Shahid, S. S. & Kale, R. B. Classical and Quantum Kernel Models for Band Gap Prediction of Bulk NiO. *Zenodo* (2026). https://doi.org/10.5281/zenodo.20963050

## License

This project is licensed under the MIT License – see the [LICENSE](LICENSE) file for details.

---

## Contact

Shahid Samad Shaikh , Prof .R.B kale
Department of Physics, Dr. Homi Bhabha State University, Institute of Science (Mumbai)  
Email: shahidshaikh010703@gmail.com
