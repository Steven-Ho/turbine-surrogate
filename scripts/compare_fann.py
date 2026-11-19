"""Collect seven faNN jobs without treating failed/incomplete models as successes."""
import argparse,csv,json
from pathlib import Path
from tfai.benchmarks.fann import MODELS,FIELDS
from tfai.benchmarks.fann_training import atomic_json,file_hash


def main():
    p=argparse.ArgumentParser(description=__doc__);p.add_argument('--root',type=Path,required=True)
    p.add_argument('--reference-root',type=Path,help='Reuse only models absent from the new root; never replace a failed new model')
    a=p.parse_args();reports={};states={};rows=[];signatures=set();scopes=set();folders={}
    for name in MODELS:
        folder=a.root/name
        if not folder.exists() and a.reference_root is not None:folder=a.reference_root/name
        folders[name]=folder
        status=folder/'status.json';report=folder/'report.json'
        states[name]=json.loads(status.read_text()) if status.exists() else {'state':'missing'}
        if states[name]['state']!='complete' or not report.exists():continue
        data=json.loads(report.read_text())
        if not (folder/'best.pt').exists() or file_hash(folder/'best.pt')!=data['best_checkpoint_sha256']:
            states[name]={'state':'artifact_mismatch','result_directory':str(folder.resolve())};continue
        states[name]['result_directory']=str(folder.resolve());reports[name]=data
        signatures.add(data['dataset_signature']);scopes.add(data['scope'])
        for i,field in enumerate(FIELDS):
            rows.append({'model':name,'field':field,'unit':data['test']['units'][i],
                         'test_rmse':data['test']['rmse'][i],'test_mae':data['test']['mae'][i],
                         'test_r2':data['test']['r2'][i],'validation_rmse':data['validation']['rmse'][i],
                         'completed_epochs':data['completed_epochs'],'best_epoch':data['best_epoch'],
                         'test_nodes':data['test']['nodes'],'test_cases':data['test']['cases']})
    compatible=len(signatures)<=1 and len(scopes)<=1
    report={'complete':len(reports)==len(MODELS) and compatible,'comparable':compatible,'states':states,'models':reports}
    atomic_json(report,a.root/'comparison.json')
    if rows:
        with (a.root/'metrics.csv').open('w') as stream:
            writer=csv.DictWriter(stream,fieldnames=list(rows[0]));writer.writeheader();writer.writerows(rows)
    if rows and compatible:
        import matplotlib
        matplotlib.use('Agg')
        import matplotlib.pyplot as plt
        fig,axes=plt.subplots(4,2,figsize=(12,14),layout='constrained')
        for ax,name in zip(axes.flat,MODELS):
            history=folders[name]/'history.json'
            ax.set_title(name);ax.set_xlabel('Epoch');ax.set_ylabel('Normalized MSE')
            if history.exists():
                points=json.loads(history.read_text())
                ax.plot([x['epoch'] for x in points],[x['train_loss'] for x in points],label='train')
                ax.plot([x['epoch'] for x in points],[x['val_loss'] for x in points],label='validation')
                ax.set_yscale('log');ax.legend();ax.grid(alpha=.25)
        axes.flat[-1].set_visible(False)
        fig.savefig(a.root/'learning_curves.png',dpi=140);plt.close(fig)
        fig,axes=plt.subplots(2,3,figsize=(14,8),layout='constrained')
        for i,(ax,field) in enumerate(zip(axes.flat,FIELDS)):
            names=[name for name in MODELS if name in reports]
            ax.bar(names,[reports[name]['test']['rmse'][i] for name in names])
            ax.set_title(field);ax.set_ylabel(reports[names[0]]['test']['units'][i]);ax.tick_params(axis='x',rotation=50)
            ax.grid(axis='y',alpha=.25)
        fig.savefig(a.root/'test_rmse.png',dpi=140);plt.close(fig)
    lines=['# faNN 七模型结果','', '数据范围：'+(', '.join(scopes) if scopes else '尚无完成结果'),'',
           '| 模型 | 状态 | 完成 epoch | 最佳 epoch |','|---|---|---:|---:|']
    for name in MODELS:
        data=reports.get(name,{})
        lines.append(f"| {name} | {states[name]['state']} | {data.get('completed_epochs','—')} | {data.get('best_epoch','—')} |")
    lines+=['','六变量原始节点验证/测试 RMSE、MAE、R² 见 metrics.csv；不得把不同单位的 RMSE 直接相加排名。',
            '各模型按完整验证集的标准化 MSE 选优，测试集在选择结束后评估；POD 的投影截断误差、UNet 的映射误差均计入原始目标误差。',
            '域范围和数据签名不同的结果不可合并比较；未完成或失败的模型不会被计作成功。']
    if rows and compatible:lines+=['','![学习曲线](learning_curves.png)','','![测试 RMSE](test_rmse.png)']
    if not compatible:lines+=['','错误：模型的数据签名或域范围不一致。']
    (a.root/'REPORT.md').write_text('\n'.join(lines)+'\n')
    if not report['complete']:raise SystemExit(1)

if __name__=='__main__':main()
