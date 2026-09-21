import earthaccess


print("Testing NASA Earthdata authentication...")

auth = earthaccess.login(
    strategy="environment"
)

if auth.authenticated:
    print("NASA Earthdata authentication successful!")
else:
    print("Authentication failed.")