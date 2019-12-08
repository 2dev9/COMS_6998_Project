import csv

# first, we import the relevant modules from the NLTK libra
reader = csv.DictReader(open("data/india-news-headlines.csv", "r", newline=''))
writers = csv.DictWriter(open("data/all_extracted.csv", "w", newline=''), fieldnames=["publish_date", "headline_text", "is_kashmir","is_pak"])
writer.writeheader()

k = ["Kashmir", "J&K", "J-K", "Hurriyat", "PoK", "LoC"]
p = ["Pak", "PoK"]

for row in reader:
	kash = False
	pak = False
	for kw in k:
		if kw in row['headline_text'] or "city.jammu"==row["headline_category"]:
			kash = True
			break
	for pw in p:
		if w in row['headline_text'] or "pak" in row["headline_category"]:
			pak = True
			break
	row.update({"is_kashmir":kash, "is_pakistan":pak})
	writer.writerow(row)

