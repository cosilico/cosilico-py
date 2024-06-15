
echo "Downloading Xenium data"
mkdir -p test_datasets/xenium
curl --output-dir test_datasets/xenium -O https://cf.10xgenomics.com/samples/xenium/2.0.0/Xenium_V1_human_Breast_2fov/Xenium_V1_human_Breast_2fov_outs.zip
unzip -o -d test_datasets/xenium/Xenium_V1_human_Breast_2fov_outs test_datasets/xenium/Xenium_V1_human_Breast_2fov_outs.zip
