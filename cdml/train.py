import os
import copy
from collections import deque
import numpy as np
import cv2
import torch
import torch.optim as optim
from tqdm import tqdm
import tensorflow as tf
import tensorboard as tb
from . import cdml

def train_no_cdml(data_streams, writer, max_steps, n_class, lr,
                  loss='triplet',
                  model_path='model', model_save_interval=2000,
                  tsne_test_interval=1000, n_test_data=1000, pretrained=False,
                  device=torch.device("cuda" if torch.cuda.is_available() else "cpu"),
                  start_point=0):
    stream_train, stream_train_eval, stream_test = data_streams
    epoch_iterator = stream_train.get_epoch_iterator()
    test_data = deque(maxlen=n_test_data)
    test_label = deque(maxlen=n_test_data)
    test_img = deque(maxlen=n_test_data)

    tri = cdml.TripletBase(n_class=n_class, pretrained=pretrained).to(device)
    optimizer_c = optim.Adam(tri.parameters(), lr=lr, weight_decay=5e-3)

    img_mean = np.array([123, 117, 104], dtype=np.float32).reshape(1, 3, 1, 1)
    cnt = 0

    tf.io.gfile = tb.compat.tensorflow_stub.io.gfile

    if start_point!=0:
        tri.load_state_dict(torch.load(os.path.join(model_path, 'model_%d.pth' % start_point)))
        cnt = start_point

    with tqdm(total=max_steps, initial=start_point) as pbar:
        for batch in copy.copy(epoch_iterator):
            x_batch, label = batch
            x_batch -= img_mean
            pbar.update(1)

            jm, _, embedding_z = tri(torch.from_numpy(x_batch).to(device))

            optimizer_c.zero_grad()
            jm.backward()
            optimizer_c.step()

            pbar.set_description("Jm: %f" % jm.item())

            if cnt > 0 and cnt % model_save_interval == 0:
                torch.save(tri.state_dict(), os.path.join(model_path, 'model_%d.pth' % cnt))

            if cnt-start_point > 0 and n_test_data > 0 and cnt % tsne_test_interval == 0:
                writer.add_embedding(np.vstack(test_data), np.vstack(test_label).flatten(),
                                     torch.from_numpy(np.stack(test_img, axis=0)),
                                     global_step=cnt, tag='embedding/train')
                writer.flush()

            writer.add_scalar('Loss/Jm/train', jm.item(), cnt)

            test_data.extend(embedding_z.detach().cpu().numpy())
            test_label.extend(label)
            for x in x_batch:
                xx = cv2.resize(x.transpose(1, 2, 0), (32, 32)).transpose(2, 0, 1)
                test_img.append((xx + img_mean[0]) / 255.0)
            cnt += 1
            if cnt > max_steps:
                break

def train_fc(data_streams, writer, max_steps, n_class,  lr_init,
                    lr_s=1.0e-3,
                  loss='triplet',
                  model_path='model', model_save_interval=500,
                  tsne_test_interval=1000, n_test_data=1000, pretrained=False,
                  device=torch.device("cuda" if torch.cuda.is_available() else "cpu"),
                  start_point=0):
    stream_train, stream_train_eval, stream_test = data_streams
    epoch_iterator = stream_train.get_epoch_iterator()
    test_data = deque(maxlen=n_test_data)
    test_label = deque(maxlen=n_test_data)
    test_img = deque(maxlen=n_test_data)

    tri = cdml.TripletFC(n_class=n_class, pretrained=pretrained).to(device)
    optimizer_c = optim.Adam(tri.embedding.parameters(),
                             lr=lr_init, weight_decay=5e-3)
    optimizer_s = optim.Adam(tri.softmax_classifier.parameters(), lr=lr_s)

    img_mean = np.array([123, 117, 104], dtype=np.float32).reshape(1, 3, 1, 1)
    cnt = 0

    tf.io.gfile = tb.compat.tensorflow_stub.io.gfile

    if start_point!=0:
        tri.load_state_dict(torch.load(os.path.join(model_path, 'model_fc_d.pth' % start_point)))
        cnt = start_point

    with tqdm(total=max_steps, initial=start_point) as pbar:
        for batch in copy.copy(epoch_iterator):
            x_batch, label = batch
            x_batch -= img_mean
            pbar.update(1)

            jclass, _, embedding_z = tri(torch.from_numpy(x_batch).to(device),
                                         torch.from_numpy(label.astype(np.int64)).to(device))

            optimizer_c.zero_grad()
            optimizer_s.zero_grad()
            jclass.backward()
            optimizer_c.step()
            optimizer_s.step()

            pbar.set_description("Jclass: %f" % jclass.item())

            if cnt > 0 and cnt % model_save_interval == 0:
                torch.save(tri.state_dict(), os.path.join(model_path, 'model_fc_%d.pth' % cnt))

            if cnt-start_point > 0 and n_test_data > 0 and cnt % tsne_test_interval == 0:
                writer.add_embedding(np.vstack(test_data), np.vstack(test_label).flatten(),
                                     torch.from_numpy(np.stack(test_img, axis=0)),
                                     global_step=cnt, tag='embedding/train')
                writer.flush()

            writer.add_scalar('Loss/Jclass/train', jclass, cnt)

            test_data.extend(embedding_z.detach().cpu().numpy())
            test_label.extend(label)
            for x in x_batch:
                xx = cv2.resize(x.transpose(1, 2, 0), (32, 32)).transpose(2, 0, 1)
                test_img.append((xx + img_mean[0]) / 255.0)
            cnt += 1
            if cnt > max_steps:
                break


def train_cdml(data_streams, writer, max_steps, n_class, lr_init,
                       lr_s=1.0e-2,
                       loss='triplet',
                       model_path='model', model_save_interval=2000,
                       tsne_test_interval=1000, n_test_data=1000, pretrained=False,
                       device=torch.device("cuda" if torch.cuda.is_available() else "cpu"),
                       start_point=0, alpha=0.25, beta=1, n_cand=4, batch=60):
    stream_train, stream_train_eval, stream_test = data_streams
    epoch_iterator = stream_train.get_epoch_iterator()
    test_data = deque(maxlen=n_test_data)
    test_label = deque(maxlen=n_test_data)
    test_img = deque(maxlen=n_test_data)

    tf.io.gfile = tb.compat.tensorflow_stub.io.gfile

    cdml_net = cdml.TripletCDML(n_class=n_class, n_cand=n_cand, alpha=alpha, beta=beta, pretrained=pretrained, batch=batch).to(device)

    optimizer_c = optim.Adam(cdml_net.embedding.parameters(),
                             lr=lr_init, weight_decay=5e-3)
    optimizer_s = optim.Adam(cdml_net.softmax_classifier.parameters(), lr=lr_s)

    jclass = 1.0e+6
    cnt = 0

    if start_point!=0:
        checkpoint = torch.load(os.path.join(model_path, 'model_alpha%.2f_cand%d_epoch%d.pth' % (alpha, n_cand, start_point)))
        cdml_net.load_state_dict(checkpoint['model_state_dict'])
        optimizer_c.load_state_dict(checkpoint['optimizer_c_state_dict'])
        optimizer_s.load_state_dict(checkpoint['optimizer_s_state_dict'])
        cnt = checkpoint['epoch']
        jclass = checkpoint['jclass']

    img_mean = np.array([123, 117, 104], dtype=np.float32).reshape(1, 3, 1, 1)

    with tqdm(total=max_steps, initial=start_point) as pbar:
        for batch in copy.copy(epoch_iterator):
            x_batch, label = batch
            x_batch -= img_mean
            pbar.update(1)

            jmetric, jclass, jcand, embedding_z = cdml_net(torch.from_numpy(x_batch).to(device),
                                                          torch.from_numpy(label.astype(np.int64)).to(device),
                                                          jclass)



            if cnt % 2 == 0:
                optimizer_c.zero_grad()
                jmetric.backward()
                optimizer_c.step()
            else:
                optimizer_s.zero_grad()
                jclass.backward()
                optimizer_s.step()



            jmetric = max(jmetric.item(), 1.0e-6)
            jclass = max(jclass.item(), 1.0e-6)

            pbar.set_description("Jmetric: %f, Jclass: %f, Jcand: %f" % (jmetric, jclass, jcand.item()))

            if cnt > 0 and cnt % model_save_interval == 0:
                torch.save({'epoch': cnt,
                            'model_state_dict': cdml_net.state_dict(),
                            'optimizer_c_state_dict': optimizer_c.state_dict(),
                            'optimizer_s_state_dict': optimizer_s.state_dict(),
                            'jclass': jclass},
                            os.path.join(model_path, 'model_alpha%.2f_cand%d_epoch%d.pth' % (alpha, n_cand, cnt)))



            if cnt-start_point > 0 and n_test_data > 0 and cnt % tsne_test_interval == 0:
                writer.add_embedding(np.vstack(test_data), np.vstack(test_label).flatten(),
                                     torch.from_numpy(np.stack(test_img, axis=0)),
                                     global_step=cnt, tag='embedding/train')
                writer.flush()


            writer.add_scalar('Loss/Jclass/train', jclass, cnt)
            writer.add_scalar('Loss/Jmetric/train', jmetric, cnt)

            test_data.extend(embedding_z.detach().cpu().numpy())
            test_label.extend(label)
            for x in x_batch:
                xx = cv2.resize(x.transpose(1, 2, 0), (32, 32)).transpose(2, 0, 1)
                test_img.append((xx + img_mean[0]) / 255.0)
            cnt += 1
            if cnt > max_steps:
                break