#%%
import ftplib
import time
import zipfile

# %%
server = "stetson.phys.dal.ca"
filedir = "Aaron/V4NA02/Monthly/ASCII/SO4/"
user = "anonymous"
password = "anonymous"
destination_folder = "data/SO4"
prefix = "GWRwSPEC_SO4_NA"
interval = 1.0  # seconds


#%%
ftp = ftplib.FTP(server)
ftp.login(user, password)

#%%
ftp.cwd(filedir)
filelist = [
    f for f in ftp.nlst()
    if f[:len(prefix)] == prefix
]
print(f"{len(filelist)} files to be downloaded")

# %%
for f in filelist:
    with open(f"{destination_folder}/{f}", 'wb') as io:
        ftp.retrbinary(f"RETR {f}", io.write)
    print(f"Downloaded {f}")
    time.sleep(interval)

# %%
for f in filelist:
    zpath = f"{destination_folder}/{f}"
    with zipfile.ZipFile(zpath, 'r') as z:
        z.extractall(zpath.replace(".zip", ""))
    print(f"extracted {zpath}")

# %%
