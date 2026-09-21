import cdsapi

print("Connecting to Copernicus CDS...")

client = cdsapi.Client()

print("CDS connection successful!")
print(client)