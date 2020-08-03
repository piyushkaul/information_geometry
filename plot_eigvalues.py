import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
sns.set(style="darkgrid")

f, axes = plt.subplots(4, 1, figsize=(5, 10), sharex=False)

for layer in range(4):
    with open('eigval_arrays'+str(layer) +'.csv','r') as fp:
        df = pd.read_csv(fp, header=0, index_col=0)
        df["id"] = df.index
        print(df)
        #pd.wide_to_long(df, ["M", "N"], i="id", j="year")
        #df.plot()
        df = df.reset_index(level='epoch')
        new_df = pd.wide_to_long(df, stubnames=["COL",], i='id', j='EIG')
        new_df = new_df.reset_index(level='EIG')
        num_rows = new_df.shape[0]
        frac=0.01
        if frac*num_rows < 1000:
            new_num_rows = num_rows
        else:
            new_num_rows = int(frac*num_rows)

        new_df = new_df.sample(n=new_num_rows,axis=0)
        print(new_df)

        new_df = new_df.rename(columns={"COL": "value", "EIG": "singular values number"})

        #ax1 = plt.subplot(4,1,layer+1)
        #axis_used = axes[layer%2, layer//2]
        axis_used = axes[layer]
        sns.lineplot(x="singular values number", y="value",
                    hue="epoch",
                    data=new_df, ax=axis_used)#kind="line"
        axis_used.set(yscale="log")
        #plt.ylim(bottom=np.power(0.1,20))

        axis_used.set_title("Singular Values for Layer " + str(layer))
plt.tight_layout()

plt.show()

    #for col in df.columns:
    #    print('type={}, val={}'.format(type(col),col))