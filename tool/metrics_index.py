from sklearn import metrics

def cluster_performance(ad, cluster_method, slice_name):

    real_y = ad.obs['ground_truth'][~ad.obs['ground_truth'].isna()]
    pre_y = ad.obs[cluster_method][~ad.obs['ground_truth'].isna()]
    # pre_y = [str(label) for label in pre_y]

    ari_score = metrics.adjusted_rand_score(real_y, pre_y)
    nmi_score = metrics.normalized_mutual_info_score(real_y, pre_y)
    ac_score = metrics.accuracy_score(real_y, pre_y)

    # jac_score = metrics.jaccard_score(real_y, pre_y, average='macro')
    print("{} | ARI:{:.4f}, NMI:{:.4f}, AC:{:.4f}".format(slice_name, ari_score, nmi_score, ac_score))
    return ari_score, nmi_score, ac_score



def evaluate_cluster_performance(adata, cluster_method='louvain'):
    print(f"=====Metrics for {cluster_method}=====")
    accuracy_list = {'all': {}}
    ari_score, nmi_score, ac_score = cluster_performance(adata, cluster_method,
                                                                     slice_name='all')
    accuracy_list['all']['ARI'] = ari_score
    accuracy_list['all']['NMI'] = nmi_score
    accuracy_list['all']['AC'] = ac_score

    for slice_name in list(adata.obs['slice_name'].unique()):
        temp_ad = adata[adata.obs['slice_name'] == slice_name]
        ari_score, nmi_score = cluster_performance(temp_ad, cluster_method, slice_name=slice_name)
        
        accuracy_list[slice_name] = {}
        accuracy_list[slice_name]['ARI'] = ari_score
        accuracy_list[slice_name]['NMI'] = nmi_score
        accuracy_list[slice_name]['AC'] = ac_score

    return accuracy_list