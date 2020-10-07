This is a tutorial for running Ubermag simulations on the remote cluster. 
1. Connect to the remote server. In terminal type 'ssh user@atom.mit.edu -p2866' where 'user' is replaced with your username
      
2. In the remote server, navigate to the directory the simulations will be run in (should have ubermag, anaconda and other modules installed). Use 'cd dirname' and 'ls' to change to directories and see what the subdirectories are in current directory.

3. Activate the environment with the desired modules with 'conda activate envname' where envname is the name of your environment. 

4. Use slurm to start the simulations. Type 'sbatch gpu_Mfield_job.sh' which calls the .sh file. Within the .sh file are the actualy instructions to run the code. 

5. Check on status of the simulation by typing 'squeue' into terminal. 

6. When the simulations are done, you can copy the simulations to your personal computer using this command run from a new terminal on your local computer: 'scp -P2866 user@atom.mit.edu:|source file| |path destination file path|' where the source file path is the file path on the server and the destination file path is the path on your local computer. 

7. Now that the simulations are on your computer, you can more easily analyze them in Jupyter and with the UbermagSANS.py library. 
