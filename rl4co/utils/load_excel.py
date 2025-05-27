from openpyxl import load_workbook
import pandas as pd

def load_data(action,reward,path,instance,algorithm):
    # 打开现有的 Excel 文件
    file_path = "/home/zhangrenyuan/demo/first/charge.xlsx"
    sheet_name = "Sheet1"  # 工作表名称
    df = pd.read_excel(file_path, sheet_name=sheet_name)


    # 定义要写入的数据
    row = None  # 指定行号
    for i in range(df["Instance"].size):
        if instance==df["Instance"][i]:
            row = i
            break


    # 将数据写入指定单元格
    name="Our"
    df[name+"-cost"][row]=-reward.item()
    df[name+"-solution"][row]=str(action.squeeze().tolist())
    df[name+"-energy"][row]=path.item()

    # 保存文件
    df.to_excel(file_path, sheet_name=sheet_name, index=False)