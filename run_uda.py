import datetime
import itertools
import torch.nn.functional as F
import torch.optim as optim
import torch.utils.data as Data
from torch.cuda import amp
import models.uda as Att
from meter import AverageMeter
from parsingsource_token import *
from process_uda import *
from tool.imblearn.over_sampling import RandomOverSampler
from tools_token import *
from code_dataset import *

os.environ["CUDA_VISIBLE_DEVICES"] = '0'

# Adjustable parameter
REGENERATE = False
dump_data_path = 'data/balanced_dump_data/'
result_file_name = 'TCNN Result'
LOOP_SIZE = 1  # 20
max_lengths = {
    'ant': 500,
    'camel': 900,
    'ivy': 1500,
    'jedit': 2500,
    'log4j': 1200,
    'lucene': 1500,
    'poi': 1800,
    'synapse': 1200,
    'xalan': 2000,
    'xerces': 2000
}
opt_num_heads = [1, 2, 4, 8]
opt_drop_rate = [0]
opt_batchsize = [8]
opt_epoch = [50]  # 15
opt_learning_rate = [1e-4]
opt_depth = [1]
opt_embedding = [40]
opt_lambda = [0.5]
# Fixed parameter
IMBALANCE_PROCESSOR = RandomOverSampler()  # RandomOverSampler(), RandomUnderSampler(), None, 'cost'
HANDCRAFT_DIM = 20
DEVICE = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
root_path_source = './data/projects/'
root_path_csv = './data/promise_data/'
package_heads = ['org', 'gnu', 'bsh', 'javax', 'com', 'fr']

# Start Time
start_time = datetime.datetime.now()
start_time_str = start_time.strftime('%Y-%m-%d_%H.%M.%S')

# Get a list of source and target projects
path_train_and_test = []
with open('data/WPDP.txt', 'r') as file_obj:
    for line in file_obj.readlines():
        line = line.strip('\n')
        line = line.strip(' ')
        path_train_and_test.append(line.split(','))

# Loop each pair of combinations
for path in path_train_and_test:
    # Get file
    path_train_source = root_path_source + path[0].split("-")[0] + "/{}".format(path[0])
    path_train_handcraft = root_path_csv + path[0].split("-")[0] + "/{}".format(path[0].split("-")[1]) + '.csv'
    path_test_source = root_path_source + path[1].split("-")[0] + "/{}".format(path[1])
    path_test_handcraft = root_path_csv + path[1].split("-")[0] + "/{}".format(path[1].split("-")[1]) + '.csv'

    # Regenerate token or get from dump_data
    print(path[0] + "===" + path[1])
    train_project_name = path_train_source
    test_project_name = path_test_source
    path_train_and_test_set = dump_data_path + path[0] + '_to_' + path[1]

    # If you don't need to regenerate, get it directly from dump_data
    if os.path.exists(path_train_and_test_set) and not REGENERATE:
        obj = load_data(path_train_and_test_set)
        [train_ast, train_label, test_ast, test_label, vocabulary_size] = obj
    else:
        # Get a list of instances of the training and test sets
        train_file_instances = extract_handcraft_instances(path_train_handcraft)
        test_file_instances = extract_handcraft_instances(path_test_handcraft)

        # Get tokens
        dict_token_train = parse_source(path_train_source, train_file_instances, package_heads)
        dict_token_test = parse_source(path_test_source, test_file_instances, package_heads)

        # Turn tokens into numbers
        list_dict, vocabulary_size = transform_token_to_number([dict_token_train, dict_token_test])
        dict_encoding_train = list_dict[0]
        dict_encoding_test = list_dict[1]

        # Take out data that can be used for training
        train_ast, train_label = extract_data(path_train_handcraft, dict_encoding_train)
        test_ast, test_label = extract_data(path_test_handcraft, dict_encoding_test)

        # Imbalanced processing
        train_ast, train_label = imbalance_process(train_ast, train_label,
                                                   IMBALANCE_PROCESSOR)

        # Saved to dump_data
        obj = [train_ast, train_label, test_ast, test_label, vocabulary_size]
        dump_data(path_train_and_test_set, obj)

    # Data from numpy to tensor
    train_ast = torch.Tensor(train_ast).to(DEVICE)
    test_ast = torch.Tensor(test_ast).to(DEVICE)

    # 数据集
    train_dataset = CodeDataset(train_ast, torch.Tensor(train_label).to(DEVICE))
    test_dataset = CodeDataset(test_ast, torch.Tensor(test_label).to(DEVICE))
    # nn_params['BATCH_SIZE'] = len(train_ast)  # 用full size batch

    loss1_meter = AverageMeter()
    loss2_pse_meter = AverageMeter()
    f1_meter = AverageMeter()
    f2_pse_meter = AverageMeter()
    f2_meter = AverageMeter()

    for params_i in itertools.product(opt_batchsize, opt_epoch, opt_learning_rate, opt_drop_rate, opt_num_heads,
                                      opt_embedding, opt_lambda, opt_depth):

        # Select nn parameters
        nn_params = {'DICT_SIZE': vocabulary_size + 1, "L2_WEIGHT": 0.005,
                     "max_token": max_lengths[path[1].split("-")[0]], "batch_size": params_i[0], 'N_EPOCH': params_i[1],
                     'LEARNING_RATE': params_i[2], "drop_rate": params_i[3], "num_head": params_i[4],
                     "EMBED_DIM": params_i[5], "lambda": params_i[6], "depth": params_i[7]}

        print("{}".format(nn_params))

        # ------------------ uda pre-training begins ------------------
        train_loader = Data.DataLoader(dataset=train_dataset, batch_size=nn_params["batch_size"], shuffle=True)
        test_loader = Data.DataLoader(dataset=test_dataset, batch_size=nn_params["batch_size"], shuffle=True)

        scaler = amp.GradScaler()
        criterion = nn.CrossEntropyLoss()
        model = Att.uda(nn_params)
        model.to(DEVICE)

        metric = []

        pre_metric = pretrain(nn_params, optim, model, nn_params['LEARNING_RATE'], train_loader, test_loader,
                              path, "./pretraining_model")
        best_p, best_r, best_f = pre_metric
        model.load_state_dict(torch.load("./pretraining_model/{}_{}".format(path[0], path[1])))

        # ------------------ uda evaluate begins ------------------

        for epoch in range(nn_params['N_EPOCH']):

            train_loader = Data.DataLoader(dataset=train_dataset, batch_size=nn_params["batch_size"], shuffle=True)
            test_loader = Data.DataLoader(dataset=test_dataset, batch_size=nn_params["batch_size"], shuffle=True)

            feat_1, feat_2, label_1, label_2 = feat_update(model, train_loader, test_loader, len(train_ast),
                                                           len(test_ast), nn_params["EMBED_DIM"])

            test_f = f1_score(test_label, label_2)
            if test_f > best_f:
                best_f = test_f
                best_p = precision_score(test_label, label_2)
                best_r = precision_score(test_label, label_2)

            # ------------------ uda training begins ------------------

            knnidx_topk = compute_knn_idx(feat_1, feat_2)

            del train_loader

            train_loader = generate_new_dataset(knnidx_topk, train_dataset, test_dataset, label_1, label_2,
                                                len(train_ast), len(test_ast), )
            model.train()
            optimizer = optim.Adam(model.parameters(), lr=nn_params['LEARNING_RATE'], betas=(0.9, 0.999),
                                   eps=1e-8,
                                   weight_decay=nn_params['L2_WEIGHT'], amsgrad=False)
            for n_iter, (asts, label, idx) in enumerate(train_loader):
                s_ast = asts[0].cuda()
                t_ast = asts[1].cuda()  # target img
                s_label = label[0].cuda()
                t_label = label[1].cuda()
                s_idx, t_idx = idx
                t_pseudo = label_2[t_idx].cuda()
                optimizer.zero_grad()
                with amp.autocast(enabled=True):
                    def distill_loss(teacher_output, student_out):
                        teacher_out = F.softmax(teacher_output, dim=-1)
                        loss = torch.sum(-teacher_out * F.log_softmax(student_out, dim=-1), dim=-1)
                        return loss.mean()


                    (feat1, cls1), (feat_2, cls2) = model(s_ast.long(), t_ast.long())
                    pre1, pre2 = torch.max(cls1, 1)[1], torch.max(cls2, 1)[1]

                    f1 = f1_score(s_label.cpu(), pre1.cpu())
                    f2 = f1_score(t_label.cpu(), pre2.cpu())
                    f2_pse = f1_score(t_pseudo.cpu(), pre2.cpu())
                    loss1 = criterion(cls1, s_label)
                    loss2 = criterion(cls2, t_pseudo)
                    loss = nn_params["lambda"] * loss2 + (1 - nn_params["lambda"]) * loss1
                scaler.scale(loss).backward()
                scaler.unscale_(optimizer)
                torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
                scaler.step(optimizer)
                scaler.update()

                loss1_meter.update(loss1)
                loss2_pse_meter.update(loss2)
                f1_meter.update(f1)
                f2_meter.update(f2)
                f2_pse_meter.update(f2_pse)

            print("Epoch[{}] Iteration[{}/{}] Loss1: {:.3f}, Loss2: {:.3f}, F1:{:.3f} ,F2:{:.3f}, F2_pse:{:.3f}  "
                  .format(epoch, (n_iter + 1), len(train_loader), loss1_meter.avg, loss2_pse_meter.avg, f1_meter.avg,
                          f2_meter.avg, f2_pse_meter.avg))
        with open("./result/WPDP_result", "a+") as f:
            f.write("{}_{}: {}\n".format(path[0], path[1], nn_params))
            f.write(
                " Pretraining-p:{:.3f}, Pretraining-r:{:.3f}, Pretraining-f:{:.3f} \n".format(
                    pre_metric[0],
                    pre_metric[1],
                    pre_metric[2]))
            f.write("p:{:.3f}, r:{:.3f}, f:{:.3f}\n".format( best_p, best_r,best_f))
# End Time
end_time = datetime.datetime.now()
print(end_time - start_time)
