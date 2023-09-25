# Meta-learning for multi-task MRI reconstruction

### Structure
```
run_fastMRI
┣ 📂functions
┣ adapt_fastMRI_few-shot.py: few-shot adaptation on target domain (fastMRI)
┣ adapt_stanford_few-shot.py: few-shot adaptation on target domain (stanford)
┣ evel_init_inference.ipynb: evaluate the initial performance of models and inference visualization
┣ maml_nmse_T8knee.py: MAML train on 8 knee tasks
┣ maml_nmse_T12mix.py: MAML train on 12 mix tasks
┣ standard_NMSE.py: standard train on 8 knee tasks/12 mix tasks
┣ standard_individual_NMSE.py: standard train on one individual distribution
┣ TTT_eval_disjoint_improved.ipynb: Test-time training evaluation with input insert for disjoint MAML
┣ TTT_eval_rss.ipynb: Test-time training evaluation on RSS coarse reconstruction
┣ TTT_eval_sensmap.ipynb: Test-time training evaluation on SENSE coarse reconstruction
┣ TTT_maml(inimg-outself)_maml_l1.py: MAML for TTT with inner loop image loss outer loop self loss
┣ TTT_maml(inself_k-disjoint-out)_l1.py: MAML for TTT with disjoint inner and outer loop
┣ TTT_maml(inself_outk)_l1.py: MAML for TTT with inner loop self loss outer loop k-space sup loss
┣ TTT_maml(inself-outimg)_l1.py: MAML for TTT with inner loop self loss outer loop image loss
┣ TTT_maml1st(inself_outk)_l1.py: FOMAML for TTT with inner loop self loss outer loop k-space sup loss
┣ TTTkspace_joint_l1.py: TTT training in kspace
┗ TTTpaper_joint_l1.py: TTT original RSS coarse recon and SENSE coarse recon
```