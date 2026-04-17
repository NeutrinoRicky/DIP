$FILE = "facades"
$URL = "http://efrosgans.eecs.berkeley.edu/pix2pix/datasets/$FILE.tar.gz"
$TAR_FILE = ".\datasets\$FILE.tar.gz"
$TARGET_DIR = ".\datasets\$FILE"

if (!(Test-Path ".\datasets")) {
    New-Item -ItemType Directory -Path ".\datasets" | Out-Null
}

if (!(Test-Path $TARGET_DIR)) {
    New-Item -ItemType Directory -Path $TARGET_DIR | Out-Null
}

Write-Host "Downloading $URL dataset to $TARGET_DIR ..."

Invoke-WebRequest -Uri $URL -OutFile $TAR_FILE

tar -zxvf $TAR_FILE -C .\datasets\

Remove-Item $TAR_FILE

Get-ChildItem ".\datasets\facades\train" -Recurse -Filter *.jpg |
    Sort-Object Name |
    ForEach-Object { $_.FullName } |
    Set-Content ".\train_list.txt"

Get-ChildItem ".\datasets\facades\val" -Recurse -Filter *.jpg |
    Sort-Object Name |
    ForEach-Object { $_.FullName } |
    Set-Content ".\val_list.txt"

Write-Host "Done."