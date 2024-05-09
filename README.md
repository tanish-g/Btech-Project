##Clone the code

```bash
git clone https://github.com/tanish-g/Btech-Project.git
```

## Dataset Downloading

To download the required datasets, execute the following commands in your terminal:

```bash
!wget --header 'Host: storage.googleapis.com' --user-agent 'Mozilla/5.0 (Macintosh; Intel Mac OS X 10.15; rv:125.0) Gecko/20100101 Firefox/125.0' --header 'Accept: text/html,application/xhtml+xml,application/xml;q=0.9,image/avif,image/webp,*/*;q=0.8' --header 'Accept-Language: en-US,en;q=0.5' --referer 'https://www.kaggle.com/' --header 'DNT: 1' --header 'Upgrade-Insecure-Requests: 1' --header 'Sec-Fetch-Dest: document' --header 'Sec-Fetch-Mode: navigate' --header 'Sec-Fetch-Site: cross-site' --header 'Sec-Fetch-User: ?1' 'https://storage.googleapis.com/kaggle-competitions-data/kaggle-v2/23870/1781260/bundle/archive.zip?GoogleAccessId=web-data@kaggle-161607.iam.gserviceaccount.com&Expires=1715405428&Signature=cb88MzHTh3ThIqH3hznhu7oTMvf%2FnWye%2Fl2WZxhs9dD52lefbmZtLktLBMw1WktECThaXQs4HV%2FeHSBdp%2F6Sk%2B2jloG0fFqr5iFvT1hlOFS5gucdjf9YWbHS6YRmrKsLA9A10m6ek51CKnANpkP23Hc405O9g18w0Ti3qYsaIR0LZSEzv%2FL7xHU6wq22XJhhGhSKuu%2BmmhSIYRomKLjzMlWb1lfKdhacKtJ9%2BCSwMbB7b5Jcy8wr5XtDqcwiIxbL9kbJWh37q4Pkoc0eDFWhKMi2%2Bex0Y61UBd5nlWpUs6IK%2BKEqe%2FWL4iaq35cPDsQAFwXI7ugp%2Fg2mo4xpwwJxgQ%3D%3D&response-content-disposition=attachment%3B+filename%3Dranzcr-clip-catheter-line-classification.zip' --output-document 'ranzcr-clip-catheter-line-classification.zip'

!wget --header 'Host: storage.googleapis.com' --user-agent 'Mozilla/5.0 (Macintosh; Intel Mac OS X 10.15; rv:125.0) Gecko/20100101 Firefox/125.0' --header 'Accept: text/html,application/xhtml+xml,application/xml;q=0.9,image/avif,image/webp,*/*;q=0.8' --header 'Accept-Language: en-US,en;q=0.5' --referer 'https://www.kaggle.com/' --header 'DNT: 1' --header 'Upgrade-Insecure-Requests: 1' --header 'Sec-Fetch-Dest: document' --header 'Sec-Fetch-Mode: navigate' --header 'Sec-Fetch-Site: cross-site' --header 'Sec-Fetch-User: ?1' 'https://storage.googleapis.com/kaggle-data-sets/1217101/2032755/compressed/train_v2.csv.zip?X-Goog-Algorithm=GOOG4-RSA-SHA256&X-Goog-Credential=gcp-kaggle-com@kaggle-161607.iam.gserviceaccount.com%2F20240508%2Fauto%2Fstorage%2Fgoog4_request&X-Goog-Date=20240508T052855Z&X-Goog-Expires=259200&X-Goog-SignedHeaders=host&X-Goog-Signature=9ed7e2c984ba4214c25bd3f608945428755af1fb70a379c8bba81b6ec2b44dc9215e3fdf7098ac53edc690afb75a3258c2db6a43e67de49aac2816f78f2090eed8f624eded282370399719150a49a7ee3b7b8b79d0dc6315acbb5f0c6cda855d59fd3e5e76e95b901fac7a0380f70d1864fd7e0c287986e5f548dd0e11668e9ea94c49285833e90012f7b111731000cace8baa754018a9c6e2edd899bc6b5fd574d614ecfb59ae66718548afc4c769cdf29549955e45cf558e72ef5da82581dfad77e828ab5a9f839406d0dc647754bb4537af1738425aa6b6e63cedd5b0b51107f20b1a41418a10d0dfd26f204264a83687156e20bbec94608cf8e874d39a13' --output-document 'train_v2.csv.zip'
```

## unzip dataset 

```bash
unzip -qq ranzcr-clip-catheter-line-classification.zip
unzip -qq train_v2.csv.zip
```

##Torch Installation

```bash
pip install torch==2.0.1 torchvision==0.15.2 torchaudio==2.0.2 --index-url https://download.pytorch.org/whl/cu118
```

##Requirements Installation

```bash
pip install warmup-scheduler
pip install segmentation-models-pytorch
pip install timm
```

##Running the code

```bash
python script_segmentation.py --enet_type '3' --fold 0
python output_segmentation_mask.py --enet_type '3' --fold 0
python script_classification.py --enet_type '3' --fold 0
```



