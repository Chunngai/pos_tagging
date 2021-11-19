from plot import plot

for i in range(7, 14):
    plot(
            id2tag_file=f"outputs/{i}/{i}-id2tag.txt", 
            rst_file=f"outputs/{i}/{i}-result.txt", 
            normalize=True, 
            title=str(i), 
            precise=False, 
            save=True
            )

