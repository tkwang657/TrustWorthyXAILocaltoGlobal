filepath="datasets/2024_lar.txt"
loader=AutoTabularLoader(file_path=filepath, chunksize=50000)
loader.print_info()
print(loader.df)
loader['activity_year', 'action_taken', 'lei']

