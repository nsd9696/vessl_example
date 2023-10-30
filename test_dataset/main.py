import vessl
import shutil

vessl.init()
data_path = "/root/examples/test_datast/logo.ico"
shutil.copy(data_path, "/root/data/logo.ico")
# vessl.upload_dataset_volume_file(dataset_name="VSSLLMFLOW", source_path=data_path, dest_path="/", organization_name="lucas")
