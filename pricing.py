import boto3
import requests
import h5py

# Reuse connection (important for speed)
session = requests.Session()

# -------------------------------
# STEP 1: GET REGIONS
# -------------------------------
ec2_global = boto3.client("ec2")
regions = [r["RegionName"] for r in ec2_global.describe_regions()["Regions"]]

# -------------------------------
# FUNCTION: GET PRICING (FAST + EXACT)
# -------------------------------
def get_region_prices(region):
    url = f"https://pricing.us-east-1.amazonaws.com/offers/v1.0/aws/AmazonEC2/current/{region}/index.json"

    try:
        data = session.get(url, timeout=15).json()
    except:
        return {}

    products = data.get("products", {})
    terms = data.get("terms", {}).get("OnDemand", {})

    price_map = {}

    for sku, product in products.items():
        attr = product.get("attributes", {})

        #  EXACT AWS UI FILTERS
        if (
            attr.get("operatingSystem") != "Linux"
            or attr.get("tenancy") != "Shared"
            or attr.get("preInstalledSw") != "NA"
            or attr.get("capacitystatus") != "Used"
        ):
            continue

        instance = attr.get("instanceType")
        if not instance:
            continue

        term = terms.get(sku, {})
        for t in term.values():
            for dim in t["priceDimensions"].values():
                price_map[instance] = float(dim["pricePerUnit"]["USD"])

    return price_map


# -------------------------------
# STEP 2: CREATE HDF5
# -------------------------------
with h5py.File("aws_final_fast.h5", "w") as hdf:

    for region in regions:
        print(f"\nProcessing {region}...")

        ec2 = boto3.client("ec2", region_name=region)

        #  Load pricing for only this region
        price_map = get_region_prices(region)

        instances, vcpus, memories, prices = [], [], [], []

        next_token = None

        while True:
            response = ec2.describe_instance_types(NextToken=next_token) if next_token else ec2.describe_instance_types()

            for inst in response["InstanceTypes"]:
                itype = inst["InstanceType"]

                instances.append(itype.encode())
                vcpus.append(inst["VCpuInfo"]["DefaultVCpus"])
                memories.append(inst["MemoryInfo"]["SizeInMiB"] / 1024)

                #  O(1) lookup
                prices.append(price_map.get(itype, 0.0))

            next_token = response.get("NextToken")
            if not next_token:
                break

        grp = hdf.create_group(region)
        grp.create_dataset("instances", data=instances)
        grp.create_dataset("vcpu", data=vcpus)
        grp.create_dataset("memory", data=memories)
        grp.create_dataset("pricing", data=prices)

        print(f" {region}: {len(instances)} instances")

print("\n DONE ")
