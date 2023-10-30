import vessl

vessl.init()
data_path = "logo.ico"
vessl.upload_dataset_volume_file(dataset_name="VSSLLMFLOW", source_path=data_path, dest_path="/", organization_name="lucas")
