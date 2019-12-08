import csv

# first, we import the relevant modules from the NLTK libra
reader = csv.DictReader(open("data/india-news-headlines.csv", "r", newline=''))
writer = csv.DictWriter(open("data/all_extracted.csv", "w", newline=''), fieldnames=["publish_date", "headline_text", "is_kashmir","is_pakistan"])
writer.writeheader()

k = ["Kashmir", "J&K", "J-K", "Hurriyat", "PoK", "LoC"]
p = ["Pak", "PoK"]

for row in reader:
	if not ("sport" in row["headline_category"] or "entertain" in row["headline_category"]):

		kash = "city.jammu"==row["headline_category"]
		pak = "world.pakistan" in row["headline_category"]

		if not kash:
			for kw in k:
				if kw in row['headline_text']:
					kash = True
					break

		if not pak:
			for pw in p:
				if pw in row['headline_text'] or "pak" in row["headline_category"]:
					pak = True
					break
		if pak or kash:
			writer.writerow({"publish_date": row["publish_date"], "headline_text":row["headline_text"],
				"is_kashmir":kash, "is_pakistan":pak})

