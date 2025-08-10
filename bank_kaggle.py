import pandas as pd
import numpy as np
from scipy.stats import norm
import matplotlib.pyplot as plt
import matplotlib as mpl

mpl.rcParams['agg.path.chunksize'] = 200000

data_1=pd.read_csv("statistics/data/train.csv", index_col="id")



data_1_all_details = data_1[np.logical_or(data_1["contact"]=="cellular", data_1["contact"]=="telephone")]

data_1_all_details["Home"]= data_1_all_details["housing"].replace({"yes":1,"no":0})
data_1_all_details["Loan"]= data_1_all_details["loan"].replace({"yes":1,"no":0})

clean_data = data_1_all_details.drop(["housing","loan"], axis=1)



q1=np.quantile(clean_data["age"], 0)

data_1_all_details_housing_1= data_1_all_details[clean_data["Home"]==1]
q2=np.quantile(data_1_all_details_housing_1["age"],0)


corr = clean_data["Home"].corr(clean_data["Loan"])
if corr >0.5:
    relation = "Considerable"
elif corr <0.5 and corr > 0.2:
    relation = "somewhat Considerable"
else:
    relation = "Not Considerable"
    
    
    
balance_home = np.logical_and(clean_data["balance"]>1500,clean_data["Home"]==1)
if balance_home.all() ==True:
    result2 = "4. If acc holder has home, then balance is more"
else:
    result2 = "4. If acc holder has home, then balance can be less or more"
    

clean_data["relationship"] = clean_data["marital"].replace({"married":1,"single":0,"divorced":0})
clean_data = clean_data.copy()

clean_data = clean_data.drop(["marital"], axis=1)

corr2 = clean_data["relationship"].corr(clean_data["Home"])

if corr2 >0.5:
    relation2 = "Considerable"
elif corr2 <0.5 and corr2 > 0.2:
    relation2 = "somewhat Considerable"
else:
    relation2 = "Not Considerable"

#clean_data.to_csv("statistics/data/cleaned_data.csv", index=True)
clean_data.reset_index(inplace=True)


clean_data_home_loan_relationship = clean_data[np.logical_and(clean_data["Home"]==1,clean_data["Loan"]==1,clean_data["relationship"]==1)]
clean_data_home_loan_relationship_0 = clean_data[np.logical_and(clean_data["Home"]==0,clean_data["Loan"]==0,clean_data["relationship"]==0)]

q3= np.quantile(clean_data_home_loan_relationship["age"], 0.2)
q4= np.quantile(clean_data_home_loan_relationship_0["age"], 0.9)


print(clean_data.head())

print("Statements:")
print(f"1. The min age to have a bank acc in this bank is {q1}")
print(f"2. {q2} is the lowest of an age of a person with acc in this bank who has a house")
print(f"3. The correlation between having a house and a loan in this bank acc users' is {relation}")
print(result2)
print(f"5. The correlation between in this bank acc users' having a house and being in diff relationship is {relation2}")
print(f"6. The back acc holder with Home and with Loan and in Relationship, 20% of them are age above {q3}")
print(f"7. The back acc holder without Home and without Loan and notin Relationship, 90% of them are age below {q4}")




sample = clean_data.sample(n=100)
sample.set_index("age", inplace=True)

clean_data_age_index = clean_data.copy()
clean_data_age_index.set_index("age", inplace=True)
fig,ax = plt.subplots(2,1)
ax[0].scatter(clean_data_age_index.index,clean_data_age_index["Home"],color="blue",label="Home")
ax[1].scatter(clean_data_age_index.index,clean_data_age_index["Loan"],color="red",label="Loan")
ax[0].set_xlabel("Age")
ax[1].set_xlabel("Age")
ax[0].legend()
ax[1].legend()
fig.savefig("statistics/data/age_home_loan.jpg")
plt.show()




fig,ax = plt.subplots(2,1)

secondary_home = np.logical_and(clean_data["Home"]==1, clean_data["education"]=="secondary").sum()
primary_home = np.logical_and(clean_data["Home"]==1, clean_data["education"]=="primary").sum()
tertiary_home = np.logical_and(clean_data["Home"]==1, clean_data["education"]=="tertiary").sum()
unknown_home = np.logical_and(clean_data["Home"]==1, clean_data["education"]=="unknown").sum()

secondary_loan = np.logical_and(clean_data["Loan"]==1, clean_data["education"]=="secondary").sum()
primary_loan = np.logical_and(clean_data["Loan"]==1, clean_data["education"]=="primary").sum()
tertiary_loan = np.logical_and(clean_data["Loan"]==1, clean_data["education"]=="tertiary").sum()
unknown_loan = np.logical_and(clean_data["Loan"]==1, clean_data["education"]=="unknown").sum()


home_list = [primary_home, secondary_home, tertiary_home, unknown_home]
loan_list = [primary_loan, secondary_loan, tertiary_loan, unknown_loan]


ax[0].pie(home_list,labels=clean_data["education"].unique(),autopct='%1.1f%%', startangle=90)
ax[0].axis('equal')  
ax[0].legend([f"Primary:{primary_home}",f"Secondary:{secondary_home}",f"Tertiary:{tertiary_home}",f"Unknown:{unknown_home}"])
ax[0].set_title("Home Ownership based on Education Level")

ax[1].pie(loan_list,labels=clean_data["education"].unique(), autopct='%1.1f%%', startangle=90)
ax[1].axis("equal")
ax[1].legend([f"Primary:{primary_loan}",f"Secondary:{secondary_loan}",f"Tertiary:{tertiary_loan}",f"Unknown:{unknown_loan}"])
ax[1].set_title("Loans taken based on Education Level")
fig.savefig("statistics/data/education_home_loan.jpg")
plt.show() 

fig,ax = plt.subplots(2,1)

ax[0].bar(clean_data["education"].unique(),home_list, color='blue', label='Home Ownership')
ax[1].bar(clean_data["education"].unique(),loan_list, color='red', label='Loan Taken')
ax[0].legend()
ax[1].legend()
ax[0].set_title("Home Ownership based on Education Level")  
ax[1].set_title("Loans taken based on Education Level")
fig.savefig("statistics/data/education_home_loan_bar.jpg")
plt.show()





fig,ax = plt.subplots(2,1)
clean_data_home = clean_data[clean_data["Home"]==1]
clean_data_loan = clean_data[clean_data["Loan"]==1]
jobs = clean_data_home["job"].unique().tolist()
ax[0].bar(jobs,clean_data_home["job"].value_counts(),color='orange')
ax[1].bar(jobs,clean_data_home["job"].value_counts(),color='green')

ax[0].set_title("Home Ownership based on Job")
ax[1].set_title("Loans taken based on Job")
fig.savefig("statistics/data/job_home_loan.jpg")
plt.show()
