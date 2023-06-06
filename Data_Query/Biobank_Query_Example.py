#File for querying the UKBioBankDataBase
#%%
import sqlite3 as sql
import pandas as pd
import random
#%%
random.seed(42) #Remember to set the seed if you need to re-extract your patients and are selecting randomly from the UKBioBank!

#%%
#Script for a use case: Wanting to extract people from the UkBioBank to create a bulkfile and clinitable, in this case we want 200 people and their
#gender 
#Bulkfiles are used with the bulkify.sh script, they tell the program what specific data you want from the
#data we want from the UKBioBank.

#Create a variable with the path to your .db file
db_file_path ="/mnt/bulk/asierrabasco/biobank/ukbiobank.db"
#Connection creation 
def create_connection(db_file):
    """ create a database connection to the SQLite database
        specified by the db_file
    :param db_file: database file
    :return: Connection object or None
    """
    conn = None
    try:
        conn = sql.connect(db_file)
    except Error as e:
        print(e)

    return conn
# %%
con= create_connection(db_file=db_file_path)
cur= con.cursor()
#Create connection and cursor objects to connect to the database
# %%
#Select which names do the tables within the database have
cur.execute('''SELECT name FROM sqlite_schema WHERE type="table" ''')
command= cur.fetchall()
#We see we have a codings and columns tables, what do they do? :)
#The other tables are called as numbers 
# %%
cur.execute('''SELECT * FROM codings''')
codies= cur.fetchall()
codies = pd.DataFrame(codies)
#The codings column has a reference to all codes within each table i.e., 
#The first column is the coding id used and the values are in the second column ie.,
# codes for 100261 are -1,1,2,3,4 and an explanation of those codes is given within the last column.
# #The linking of these codes to the actual tables are in columns, which explains the table names we 
# #see in the schema above and give us the coding used. 
# %%
cur.execute('''SELECT * FROM columns ''')
colnames = ["table_name","data_type","description","coding"]
table_exp = cur.fetchall()
table_exp = pd.DataFrame(table_exp)
table_exp.columns= colnames
# %% We now want the sex of patients with brain MRI imaging,
#Let us browse columns for sex and see which tabke best fits the sex:
img_tables=table_exp[table_exp["description"].str.contains("sex")]
#It is table 31
#%%
#The table is 20252 for MRI patients with T1, let us see the column names to get an idea of what to query

a= cur.execute('''SELECT "31".instance_index FROM "31" ''')
a=a.fetchall()
a= pd.DataFrame(a)
print(list(a[0]).count(0)) #Count if there have been gender re-recordings, if not, 0 count should match row number

#%%
#The UKBioBank has patients coming again for re-imaging. As we are interested only on their first images,
#We just select instance index=2 (As this would be the second time the patient comes, the first being when they were enrolled before the imaging period)
#The structure of every sub-table on the UKBioBank that is not codings or 
#Columns is always the same: the eid of the patient, the instance (which number of record of that patient it is) and
#The array whcih indicates to which table it belongs to. The last +#Column is always the table number, and has the information in codings format e.g, 0,1 for gender which are male-female

a = cur.execute('''SELECT "20252".eid, "20252".instance_index, "20252".array_index, "31"."31"
                    FROM "20252" 
                    LEFT JOIN "31" ON "20252".eid ="31".eid 
                    WHERE "20252".instance_index = 2 
                    ''')
a=a.fetchall()
a=pd.DataFrame(a)
#We create a nice dataframe
a.columns= ["ID","Instance","Array","Gender"]
#We cand ecodify the gender later

#%%
#With this, we can create two things: the New ID that the patient will have, this is
#output into a .bulk file which is then read by a bash script on the terminal (ask Marko for the bulkify script and components)
a= a.assign(NewID=a.ID.apply(str) + " 20252_" + a.Instance.apply(str) + "_" + a.Array.apply(str))

#The filenames variable is used to create the clinitable, as the files fetched from the UKBioBank are .zip which then can be put with 
#The table variable (Gender in this case) into a file for easy access for the experiments.
a= a.assign(FileNames=a.ID.apply(str) + "_20252_" + a.Instance.apply(str) + "_" + a.Array.apply(str))
# %%
#Now we select randomly 200 males and 200 females:

lst= a.groupby("Gender", group_keys=False).apply(lambda x: x.sample(100))[["NewID","Gender"]]

#%%
#This opens/creates and opens a file called ukbulkify which contains in a specific format that the ukbiobank program uses the iDs of the patients we want.
with open("/mnt/bulk/asierrabasco/biobank/Gender_Project_Preprocessing/ukbulkify.bulk", mode="a") as f:
    for item in list(lst["NewID"]):
        f.write("%s\n" % item)
    print("List printed")

# %%
#Output the clinitable
lst[["FileNames","Gender"]].to_csv("whatever path you want", index=False)
