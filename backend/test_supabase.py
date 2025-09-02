from supabase import create_client

url = "https://uxrgkpbgatlbrkqilury.supabase.co"
key = "sb_secret_rS1q0-u-vp2WMoD4tB_H8Q_6p2OoMvP"

supabase = create_client(url, key)

# Test: fetch all rows from Product table
data = supabase.table("products").select("*").execute()
print(data)
