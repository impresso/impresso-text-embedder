#!/bin/bash

# Check for correct number of arguments
if [ "$#" -ne 2 ]; then
    echo "Usage: $0 <newspaper_provider> <gpu_number>"
    exit 1
fi

# Arguments
PROVIDER=$1
GPU_NUMBER=$2

# Dictionary of newspaper collections
declare -A NEWSPAPER_COLLECTION
NEWSPAPER_COLLECTION["BCUL"]="ACI Castigat CL Croquis FAMDE FAN GAVi AV JY2 JV JVE JH OBS Bombe Cancoire Fronde Griffe Guepe1851 Guepe1887 RLA Charivari CharivariCH Grelot Moniteur ouistiti PDL PJ TouSuIl VVS1 MESSAGER PS NV ME MB NS FAM FAV1 EM esta PAT VVS NV1 NV2"
NEWSPAPER_COLLECTION["BNF"]="excelsior lafronde oeuvre marieclaire"
NEWSPAPER_COLLECTION["BNF-EN"]="legaulois lematin lepji lepetitparisien oecaen oerennes jdpl"
NEWSPAPER_COLLECTION["BNL"]="actionfem armeteufel avenirgdl buergerbeamten courriergdl deletz1893 demitock diekwochen dunioun gazgrdlux indeplux kommmit landwortbild lunion luxembourg1935 luxland luxwort luxzeit1844 luxzeit1858 obermosel onsjongen schmiede tageblatt volkfreu1869 waechtersauer waeschfra"
NEWSPAPER_COLLECTION["LeTemps"]="GDL JDG"
NEWSPAPER_COLLECTION["RERO"]="BDC CDV DLE EDA EXP IMP JDF JDV LBP LCE LCG LCS LCR LES LNF LSE LSR LTF LVE"
NEWSPAPER_COLLECTION["RERO2"]="BLB BNN DFS DVF EZR FZG HRV LAB LLE MGS NTS NZG SGZ SRT WHD ZBT"
NEWSPAPER_COLLECTION["RERO3"]="CON DTT FCT GAV GAZ LLS OIZ SAX SDT SMZ VDR VHT"
NEWSPAPER_COLLECTION["SWA"]="arbeitgeber handelsztg"
NEWSPAPER_COLLECTION["UZH"]="FedGazFr FedGazDe NZZ"

# Get the newspapers for the specified provider
NEWSPAPERS=${NEWSPAPER_COLLECTION[$PROVIDER]}

# Check if the provider exists
if [ -z "$NEWSPAPERS" ]; then
    echo "Provider $PROVIDER not found!"
    exit 1
fi

# Export the GPU number globally for the entire script
export CUDA_VISIBLE_DEVICES=$GPU_NUMBER

# Loop over each newspaper and run the make commands
for NEWSPAPER in $NEWSPAPERS; do
    echo "Syncing and running for newspaper: $NEWSPAPER"

    # Sync the newspaper
    make sync newspaper NEWSPAPER=$NEWSPAPER

    # Log the GPU number being used
    echo "Running on GPU number: $GPU_NUMBER (CUDA_VISIBLE_DEVICES=$CUDA_VISIBLE_DEVICES)"

    # Run the newspaper make command with GPU
    make -j 30 newspaper NEWSPAPER=$NEWSPAPER
done

make -j 30 newspaper NEWSPAPER=ACI
