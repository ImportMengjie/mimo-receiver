import abc
import torch
import numpy as np
import torch.nn.functional as F

from loader import CsiDataloader
from model import DetectionNetModel
from utils import complex2real


class DetectionMethod(abc.ABC):

    def __init__(self, modulation):
        self.modulation = modulation

    @abc.abstractmethod
    def get_key_name(self):
        pass

    @abc.abstractmethod
    def get_x_hat(self, y, h, x, var):
        pass

    def get_nmse(self, y, h, x, var):
        x_hat = self.get_x_hat(y, h, x, var)
        # x = complex2real(x)
        # x_hat = complex2real(x_hat)
        nmse = ((torch.abs(x - x_hat) ** 2).sum(-1).sum(-1) / (torch.abs(x) ** 2).sum(-1).sum(-1)).mean()
        nmse = 10 * torch.log10(nmse)
        return nmse.item()

    def get_ber(self, y, h, x, x_idx, var):
        x_hat = self.get_x_hat(y, h, x, var)
        x_hat_r = x_hat.real
        x_hat_i = x_hat.imag
        constellation = torch.from_numpy(CsiDataloader.constellations[self.modulation])
        constellation_r = constellation.real
        constellation_i = constellation.imag

        x_hat_dist = (x_hat_r - constellation_r) ** 2 + (x_hat_i - constellation_i) ** 2
        x_hat_idx = torch.argmin(x_hat_dist, dim=-1, keepdim=True)
        ber = (x_hat_idx != x_idx).sum() / x_idx.numel()
        return ber.item()


class DetectionMethodZF(DetectionMethod):

    def __init__(self, modulation):
        super().__init__(modulation)

    def get_key_name(self):
        return 'ZF'

    def get_x_hat(self, y, h, x, var):
        x_hat = torch.inverse(h.conj().transpose(-1, -2) @ h) @ h.conj().transpose(-1, -2) @ y
        return x_hat


class DetectionMethodMMSE(DetectionMethod):

    def __init__(self, modulation):
        super().__init__(modulation)

    def get_key_name(self):
        return 'MMSE'

    def get_x_hat(self, y, h, x, var):
        A = h.conj().transpose(-1, -2) @ h + var * torch.eye(h.shape[-1], h.shape[-1])
        x_hat = torch.inverse(A) @ h.conj().transpose(-1, -2) @ y
        return x_hat


class DetectionMethodModel(DetectionMethod):

    def __init__(self, model: DetectionNetModel, modulation, use_gpu):
        self.model = model.eval()
        self.use_gpu = use_gpu
        super().__init__(modulation)

    def get_key_name(self):
        return self.model.name

    def get_x_hat(self, y, h, x, var):
        A = h.conj().transpose(-1, -2) @ h + var * torch.eye(h.shape[-1], h.shape[-1])
        b = h.conj().transpose(-1, -2) @ y

        b = torch.cat((b.real, b.imag), 2)
        A_left = torch.cat((A.real, A.imag), 2)
        A_right = torch.cat((-A.imag, A.real), 2)
        A = torch.cat((A_left, A_right), 3)
        if self.use_gpu:
            A = A.cuda()
            b = b.cuda()
        x_hat, _ = self.model(A, b)  # reshape???
        x_hat = x_hat[:, :, 0:x.shape[-2], :] + x_hat[:, :, x.shape[-2]:, :] * 1j
        if x_hat.is_cuda:
            x_hat = x_hat.cpu()
        return x_hat


class DetectionMethodML(DetectionMethod):
    def get_key_name(self):
        return 'ml'

    def get_x_hat(self, y, h, x, var):
        pass


class DetectionMethodSD(DetectionMethod):

    def sphereDecoding(m, n, H, variance, QAM=4):
        INF = 1000111000111
        alpha = 2
        v = np.random.normal(0, np.sqrt(variance), (n, 1))

        s = 2 * np.random.random_integers(1, QAM, (m, 1)) - (QAM + 1)
        x = np.dot(H, s) + v

        d = alpha * variance * n
        print("Algorithm est for radius = ", np.sqrt(d))
        babaiB = np.floor(np.dot(np.linalg.pinv(H), x))
        babaiD = np.linalg.norm(x - np.dot(H, babaiB))
        print("Babai est for radius =", babaiD)
        print(QAM)
        print(s)
        res = np.linalg.qr(H)
        R = res[1]
        q1 = res[0]

        y = np.dot(q1.conj().T, x)
        _y = y.copy()

        D = np.zeros(m)
        UB = np.zeros(m)
        k = m - 1
        D[k] = np.sqrt(d)
        setUB = 1
        flopsCount = 0
        ans = INF
        answer = np.zeros(m)

        ###Start
        for _ in range(1, 10):
            k = m - 1
            _y = y.copy()
            D = np.zeros(m)
            UB = np.zeros(m)
            D[k] = np.sqrt(d)
            setUB = 1
            while True:
                flopsCount += 1
                if setUB == 1:
                    if (D[k] + _y[k]) / R[k][k] > (-D[k] + _y[k]) / R[k][k]:

                        UB[k] = np.floor((D[k] + _y[k]) / R[k][k])
                        s[k] = np.ceil((-D[k] + _y[k]) / R[k][k]) - 1
                    else:
                        UB[k] = np.floor((-D[k] + _y[k]) / R[k][k])
                        s[k] = np.ceil((D[k] + _y[k]) / R[k][k]) - 1
                    te = s[k] + 1
                    for j in range(QAM - 1, -QAM, -2):
                        if te > j:
                            break
                        s[k] = j - 2
                s[k] = s[k] + 2
                # print(k,s[k],UB[k])
                setUB = 0
                if s[k] <= UB[k] and s[k] < QAM:
                    if k == 0:
                        if ans > np.linalg.norm(np.dot(H, s) - x):
                            ans = np.linalg.norm(np.dot(H, s) - x)
                            answer = s.copy()
                            print("***", answer)
                        # print(s,np.linalg.norm(np.dot(H,s.T)-x.T) )
                    else:
                        k = k - 1
                        _y[k] = y[k]
                        for i in range(k + 1, m):
                            # flopsCount += 1
                            _y[k] -= (R[k][i] * s[i])

                        D[k] = np.sqrt(D[k + 1] ** 2 - (_y[k + 1] - R[k + 1][k + 1] * s[k + 1]) ** 2)
                        setUB = 1
                    continue
                else:
                    k = k + 1
                    if k == m:
                        break

            if ans == INF:
                print("The Radius is not big enough")
                d *= alpha
                print(np.sqrt(d))
            else:
                break

        flopsCount *= 14
        return answer

    def get_key_name(self):
        return 'SD'

    def get_x_hat(self, y, h, x, var):
        pass


class DetectionMethodConjugateGradient(DetectionMethod):

    def __init__(self, modulation, iterate, name_add_iterate=True):
        self.iterate = iterate
        self.name_add_iterate = name_add_iterate
        super().__init__(modulation)

    def get_key_name(self):
        if self.name_add_iterate:
            return 'CG-{}th'.format(self.iterate)
        else:
            return self.get_key_name_short()

    def get_key_name_short(self):
        return 'CG'

    @staticmethod
    def conjugate(s, r, d, A):
        alpha = (r.conj().transpose(-1, -2) @ r) / (r.conj().transpose(-1, -2) @ A @ d)
        s_next = s + alpha * d
        r_next = r - alpha * (A @ d)
        beta = (r_next.conj().transpose(-1, -2) @ r_next) / (r.conj().transpose(-1, -2) @ r)
        d_next = r_next + beta * d
        return s_next, r_next, d_next

    def get_x_hat(self, y, h, x, var):
        A = h.conj().transpose(-1, -2) @ h + var * torch.eye(h.shape[-1], h.shape[-1])
        b = h.conj().transpose(-1, -2) @ y

        b = torch.cat((b.real, b.imag), 2)
        A_left = torch.cat((A.real, A.imag), 2)
        A_right = torch.cat((-A.imag, A.real), 2)
        A = torch.cat((A_left, A_right), 3)

        s = torch.zeros(b.shape)
        if torch.cuda.is_available():
            s = s.cuda()
        r = b
        d = r
        for i in range(self.iterate):
            s, r, d = DetectionMethodConjugateGradient.conjugate(s, r, d, A)
        s = s[:, :, 0:x.shape[-2], :] + s[:, :, x.shape[-2]:, :] * 1j
        return s
