import http.client
import json
import os
import math
import csv

host = "ecshweb.pchome.com.tw"
headers = {
    "User-Agent": "Mozilla/5.0 (Macintosh; Intel Mac OS X 10_15_7) AppleWebKit/537.36 (KHTML, like Gecko) Chrome/131.0.0.0 Safari/537.36",
    "Referer": "https://24h.pchome.com.tw/",
    "Sec-Ch-Ua": '"Google Chrome";v="131", "Chromium";v="131", "Not_A Brand";v="24"',
    "Sec-Ch-Ua-Mobile": "?0",
    "Sec-Ch-Ua-Platform": '"macOS"',
}

page = 1
total_price = 0
product_count = 0
prices = []
products = []

output_dir = "week-1"
products_path = os.path.join(output_dir, "products.txt")
best_products_path = os.path.join(output_dir, "best-products.txt")
standardization_path = os.path.join(output_dir, "standardization.csv")

with open(products_path, "w", encoding="utf-8") as file:
    file.write("")
with open(best_products_path, "w", encoding="utf-8") as best_file:
    best_file.write("")

while True:
    path = f"/search/v4.3/all/results?cateid=DSAA31&attr=&pageCount=40&page={page}"
    conn = http.client.HTTPSConnection(host)
    conn.request("GET", path, headers=headers)
    response = conn.getresponse()

    if response.status == 200:
        data = response.read().decode("utf-8")
        data_dict = json.loads(data)

        prods = data_dict.get("Prods", [])
        if prods:
            with open(products_path, "a", encoding="utf-8") as file:
                with open(best_products_path, "a", encoding="utf-8") as best_file:
                    for prod in prods:
                        prod_id = prod.get("Id")
                        price = prod.get("Price", 0)
                        product_name = prod.get("Name", "")

                        if price > 0:
                            prices.append(price)
                            products.append({"Id": prod_id, "Price": price})

                        file.write(prod_id + "\n")

                        review_count = prod.get("reviewCount", 0)
                        review_value = prod.get("ratingValue", 0.0)
                        if isinstance(review_count, (int, float)) and isinstance(
                            review_value, (int, float)
                        ):
                            if review_count > 0 and review_value > 4.9:
                                best_file.write(prod_id + "\n")

                        if "i5處理器" in product_name:
                            total_price += price
                            product_count += 1
            page += 1
        else:
            break
    else:
        print(f"page {page} API request failed: {response.status}")
        break

    conn.close()


if prices:
    mean_price = sum(prices) / len(prices)
    std_dev_price = math.sqrt(sum((p - mean_price) ** 2 for p in prices) / len(prices))

    if std_dev_price == 0:
        z_scores = [(product["Id"], product["Price"], 0) for product in products]
    else:
        z_scores = [
            (
                product["Id"],
                product["Price"],
                (product["Price"] - mean_price) / std_dev_price,
            )
            for product in products
        ]

    with open(standardization_path, "a", encoding="utf-8", newline="") as csvfile:
        csvwriter = csv.writer(csvfile)
        csvwriter.writerows(z_scores)

if product_count > 0:
    average_price = total_price / product_count
    print(
        f"The average price of ASUS PCs with Intel i5 processor: NT${average_price:.2f}"
    )
else:
    print("No ASUS PCs with Intel i5 processor found.")