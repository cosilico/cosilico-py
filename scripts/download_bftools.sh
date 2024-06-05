mkdir -p extras/bioformats
curl --output-dir extras/bioformats -O https://downloads.openmicroscopy.org/bio-formats/7.3.0/artifacts/bftools.zip
unzip -o -d extras/bioformats extras/bioformats/bftools.zip
