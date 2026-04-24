import os

def create_linked_docs(out_dir="linked_docs"):
    os.makedirs(out_dir, exist_ok=True)
    
    docs = [
        {
            "name": "mission_orion.txt",
            "content": (
                "The Orion Mission is a strategic space exploration program led by NASA. "
                "Its primary goal is to establish a human presence on Mars by the mid-2035s. "
                "The mission relies heavily on the Space Launch System (SLS), which is the most powerful rocket ever built. "
                "The SLS rocket is designed to carry the Orion spacecraft beyond low Earth orbit."
            )
        },
        {
            "name": "rocket_sls.txt",
            "content": (
                "The Space Launch System (SLS) is a super heavy-lift launch vehicle. "
                "The core stage of the SLS rocket features four powerful RS-25 engines. "
                "The SLS rocket uses solid rocket boosters provided by Northrop Grumman to achieve initial lift-off. "
                "The RS-25 engines are critical for the sustained climb into orbit."
            )
        },
        {
            "name": "engine_rs25.txt",
            "content": (
                "The RS-25 engine, also known as the Space Shuttle Main Engine (SSME), is a high-performance liquid-fuel cryogenic rocket engine. "
                "RS-25 engines are manufactured by the American company Aerojet Rocketdyne. "
                "Aerojet Rocketdyne produces these engines at their facility in Canoga Park, California. "
                "The engines burn liquid hydrogen and liquid oxygen as propellants."
            )
        }
    ]
    
    for doc in docs:
        with open(os.path.join(out_dir, doc["name"]), "w", encoding="utf-8") as f:
            f.write(doc["content"])
    
    print(f"Created {len(docs)} linked documents in '{out_dir}/'")

if __name__ == "__main__":
    create_linked_docs()
