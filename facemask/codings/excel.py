import pandas as pd
import openpyxl



df = pd.DataFrame([[11, 21, 31], [12, 22, 32], [31, 32, 33]],
                  index=['one', 'two', 'three'], columns=['a', 'b', 'c'])
				  
print("Data Uploading..")
df.to_excel('pandas_to_excel.xlsx', sheet_name='new_sheet_name')
with pd.ExcelWriter('pandas_to_excel.xlsx') as writer:
    df.to_excel(writer, sheet_name='hello')
print("Completed")


# Adding Data on Excel sheet..
		current_time = datetime.datetime.now() 	
		df = pd.DataFrame([[current_time,count, c3, c4]],
            index=[i], columns=['Date_Time','Total_Count', 'With_Mask', 'Without_Mask'])
				  
		df.to_excel('Data.xlsx', sheet_name='new_sheet_name')
		with pd.ExcelWriter('Data.xlsx') as writer:
			df.to_excel(writer, sheet_name='sheet')
		
		print(i)