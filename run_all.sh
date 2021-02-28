for i in '40' '60' '80'
do
    for j in '1' '2' '3' '4'
    do
        sbatch run_vae_$i\_$j\.sh
    done
done
