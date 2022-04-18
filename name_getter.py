import requests as r
from bs4 import BeautifulSoup

if __name__ == "__main__":
    resp = r.get("https://en.wikipedia.org/wiki/List_of_towns_in_England")
    soup = BeautifulSoup(resp.text, features="lxml")
    
    names = []

    rows = soup.find_all(lambda tag: tag.name == "tr")
    
    for row in rows:
        left_col = row.find(lambda tag: tag.name == "td")
        if left_col is not None:
            name = left_col.text
            if "&" in name:
                name = name.replace("&", "and")
            print(name)
            names.append(name.upper())
            if name == "Welwyn Garden City":
                break
                
    print(f"Found {len(names)} town names")
    
    alphabet = set()
    for name in names:
        for letter in name:
            alphabet.add(letter.upper())
        
    alphabet = sorted(alphabet)
    print(alphabet)
    
    print(max(len(x) for x in names))
    with open("towns.txt", "w") as f:
        f.writelines(name + "\n" for name in names)