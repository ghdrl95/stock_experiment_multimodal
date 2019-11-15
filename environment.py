import numpy as np
import random
import chart_draw as cd
import PIL.Image as pilimg
import os
import json

class Environment:

    chartcode_list = [
        "000060",
        "000080",
        "000087",
        "000227",
        "000240",
        "000325",
        "000500",
        "000520",
        "000547",
        "000760",
        "000860",
        "000880",
        "000885",
        "000950",
        "000995",
        "001060",
        "001065",
        "001080",
        "001120",
        "001140",
        "001260",
        "001270",
        "001275",
        "001390",
        "001430",
        "001529",
        "001550",
        "001680",
        "001770",
        "001790",
        "001799",
        #"001800",
        "001880",
        "002020",
        "002025",
        "002100",
        "002200",
        "002210",
        "002220",
        "002240",
        "002250",
        "002300",
        "002390",
        "002460",
        "002620",
        "002795",
        "002810",
        "002995",
        "003000",
        "003075",
        "003090",
        "003220",
        "003350",
        "003490",
        "003495",
        "003960",
        "004360",
        "004380",
        "004430",
        "004545",
        "004700",
        "004960",
        "004970",
        "005070",
        "005190",
        #"005250",
        "005257",
        "005420",
        #"005440",
        "005820",
        "005850",
        "005880",
        #"005935",
        "005940",
        #"005965",
        "006120",
        "006125",
        "007340",
        "007540",
        "007570",
        "007575",
        "008355",
        "008490",
        "008500",
        "009140",
        "009420",
        "009470",
        "009770",
        "009830",
        "010660",
        "010780",
        "011155",
        "011760",
        "011785",
        "011790",
        "011810",
        "012450",
        "012630",
        "013580",
        "014280",
        #"014820",
        #"014825",
        "015760",
        "016360",
        "016385",
        "016580",
        "016800",
        "019440",
        "020150",
        "023450",
        "023590",
        "024110",
        "024720",
        "026960",
        "027390",
        #"028050",
        "028100",
        "029460",
        "029780",
        "030000",
        "030200",
        "032640",
        "033240",
        "033270",
        "034120",
        "034220",
        "034300",
        "034310",
        "034590",
        "035250",
        "042660",
        "044820",
        "047050",
        "047810",
        "051600",
        "052690",
        "055490",
        "058430",
        "063160",
        "064350",
        "066575",
        "068290",
        "069260",
        "069500",
        "069660",
        "071090",
        "078000",
        "078935",
        "079430",
        "079440",
        #"079550",
        "081000",
        "084670",
        "091160",
        "091180",
        "091230",
        "093050",
        "099140",
        "102110",
        "102460",
        "103140",
        "105190",
        "105630",
        "108590",
        "108675",
        "111770",
        "112610",
        "114090",
        "115390",
        "117690",
        "120115",
        "122090",
        "131890",
        "138540",
        "139260",
        "139290",
        "140950",
        "143850",
        "143860",
        "145850",
        "145995",
        "147970",
        "148020",
        "152100",
        "152330",
        "152870",
        "153270",
        "159800",
        "168300",
        "168580",
        "169950",
        "174360",
        "180640",
        "183710",
        "185680",
        "192720",
        "194370",
        "195870",
        "200020",
        "200030",
        "200040",
        "203780",
        #"204320",
        "210540",
        "210980",
        "211260",
        "211560",
        "213500",
        "217790",
        #"226320",
        "226490",
        "227540",
        "227830",
        "227840",
        "234080",
        "237350",
        "241560",
        "243880",
        "245340",
        "248170",
        "249420",
        "260200",
        "261220",
        "261920",
        "265690",
        "266420",
        #"267260",
        "267290",
        "272450",
        "272550",
        "275540",
        "277630",
        "281820",
        "293180",
        "293480",
        "294400",
        "302450",
        "305050",
        "306520",
        "310960",
        "315270",
        "316140",
        "500009",
        "500034",
        "510008",
        "520004",
        "520009",
        "520010",
        "520025",
        "530014",
        "550009",
        "550014",
        "550051",
        "590007",
        "590009",
    ]

    chartcode_value={
        "000060":1,
        "000080":1,
        "000087":1,
        "000227":1,
        "000240":1,
        "000325":1,
        "000500":1,
        "000520":1,
        "000547":1,
        "000760":1,
        "000860":1,
        "000880":1,
        "000885":1,
        "000950":1,
        "000995":1,
        "001060":1,
        "001065":1,
        "001080":1,
        "001120":1,
        "001140":1,
        "001260":1,
        "001270":1,
        "001275":1,
        "001390":1,
        "001430":1,
        "001529":1,
        "001550":1,
        "001680":1,
        "001770":1,
        "001790":1,
        "001799":1,
        # "00180:10",
        "001880":1,
        "002020":1,
        "002025":1,
        "002100":1,
        "002200":1,
        "002210":1,
        "002220":1,
        "002240":1,
        "002250":1,
        "002300":1,
        "002390":1,
        "002460":1,
        "002620":1,
        "002795":1,
        "002810":1,
        "002995":1,
        "003000":1,
        "003075":1,
        "003090":1,
        "003220":1,
        "003350":1,
        "003490":1,
        "003495":1,
        "003960":1,
        "004360":1,
        "004380":1,
        "004430":1,
        "004545":1,
        "004700":1,
        "004960":1,
        "004970":1,
        "005070":1,
        "005190":1,
        # "00525:10",
        "005257":1,
        "005420":1,
        # "00544:10",
        "005820":1,
        "005850":1,
        "005880":1,
        # "00593:15",
        "005940":1,
        # "00596:15",
        "006120":1,
        "006125":1,
        "007340":1,
        "007540":1,
        "007570":1,
        "007575":1,
        "008355":1,
        "008490":1,
        "008500":1,
        "009140":1,
        "009420":1,
        "009470":1,
        "009770":1,
        "009830":1,
        "010660":1,
        "010780":1,
        "011155":1,
        "011760":1,
        "011785":1,
        "011790":1,
        "011810":1,
        "012450":1,
        "012630":1,
        "013580":1,
        "014280":1,
        # "01482:10",
        # "01482:15",
        "015760":1,
        "016360":1,
        "016385":1,
        "016580":1,
        "016800":1,
        "019440":1,
        "020150":1,
        "023450":1,
        "023590":1,
        "024110":1,
        "024720":1,
        "026960":1,
        "027390":1,
        # "02805:10",
        "028100":1,
        "029460":1,
        "029780":1,
        "030000":1,
        "030200":1,
        "032640":1,
        "033240":1,
        "033270":1,
        "034120":1,
        "034220":1,
        "034300":1,
        "034310":1,
        "034590":1,
        "035250":1,
        "042660":1,
        "044820":1,
        "047050":1,
        "047810":1,
        "051600":1,
        "052690":1,
        "055490":1,
        "058430":1,
        "063160":1,
        "064350":1,
        "066575":1,
        "068290":1,
        "069260":1,
        "069500":1,
        "069660":1,
        "071090":1,
        "078000":1,
        "078935":1,
        "079430":1,
        "079440":1,
        # "07955:10",
        "081000":1,
        "084670":1,
        "091160":1,
        "091180":1,
        "091230":1,
        "093050":1,
        "099140":1,
        "102110":1,
        "102460":1,
        "103140":1,
        "105190":1,
        "105630":1,
        "108590":1,
        "108675":1,
        "111770":1,
        "112610":1,
        "114090":1,
        "115390":1,
        "117690":1,
        "120115":1,
        "122090":1,
        "131890":1,
        "138540":1,
        "139260":1,
        "139290":1,
        "140950":1,
        "143850":1,
        "143860":1,
        "145850":1,
        "145995":1,
        "147970":1,
        "148020":1,
        "152100":1,
        "152330":1,
        "152870":1,
        "153270":1,
        "159800":1,
        "168300":1,
        "168580":1,
        "169950":1,
        "174360":1,
        "180640":1,
        "183710":1,
        "185680":1,
        "192720":1,
        "194370":1,
        "195870":1,
        "200020":1,
        "200030":1,
        "200040":1,
        "203780":1,
        # "20432:10",
        "210540":1,
        "210980":1,
        "211260":1,
        "211560":1,
        "213500":1,
        "217790":1,
        # "22632:10",
        "226490":1,
        "227540":1,
        "227830":1,
        "227840":1,
        "234080":1,
        "237350":1,
        "241560":1,
        "243880":1,
        "245340":1,
        "248170":1,
        "249420":1,
        "260200":1,
        "261220":1,
        "261920":1,
        "265690":1,
        "266420":1,
        # "26726:10",
        "267290":1,
        "272450":1,
        "272550":1,
        "275540":1,
        "277630":1,
        "281820":1,
        "293180":1,
        "293480":1,
        "294400":1,
        "302450":1,
        "305050":1,
        "306520":1,
        "310960":1,
        "315270":1,
        "316140":1,
        "500009":1,
        "500034":1,
        "510008":1,
        "520004":1,
        "520009":1,
        "520010":1,
        "520025":1,
        "530014":1,
        "550009":1,
        "550014":1,
        "550051":1,
        "590007":1,
        "590009":1,
    }
    KOSPI_DATA = None
    KOSPI_DATE = None
    # 종가 데이터 위치 - 보상계산용
    PRICE_IDX = 1
    #candle stick file path
    #FILE_PATH =
    FILE_PATH = ['F:/chart_images_other_basic/', 'F:/chart_images_other_BB/', 'F:/chart_images_other_MACD/', 'F:/chart_images_other_OBV/', 'F:/chart_images_other_DMI/', 'F:/chart_images_other_STO/','F:/chart_images_other/','F:/chart_images_other_KOSPI/']
    FILE_PATH_TEST = ['F:/chart_images_other_basic_test/', 'F:/chart_images_other_BB_test/', 'F:/chart_images_other_MACD_test/',
                 'F:/chart_images_other_OBV_test/', 'F:/chart_images_other_DMI_test/', 'F:/chart_images_other_STO_test/',
                 'F:/chart_images_other_test/','F:/chart_images_other_KOSPI_test/']

    FILE_TYPE = [FILE_PATH, FILE_PATH_TEST]

    TYPE_BASIC = 0  # 캔들차트 + 이평선 + 거래량
    TYPE_BB = 1     # 볼린저 밴드
    TYPE_MACD = 2   # MACD
    TYPE_OBV = 3    # OBV
    TYPE_DMI = 4    # DMI
    TYPE_STO = 5    # 스토캐스팅
    TYPE_ORIGIN = 6
    TYPE_KOSPI = 7
    #RANGE_SHAPE = {5: [630, 130, 4], 20: [630, 245, 4], 60: [630, 550, 4], 6:[630,130,4], 7:[630,130,4], 25: [630, 280, 4]}
    RANGE_SHAPE = {20: [630, 245, 3], 5: [630, 130, 4],  60: [630, 550, 4], 6: [630, 130, 4], 7: [630, 130, 4],
                   25: [630, 280, 3]}
    RANGE_SHAPES = {20: [ [400, 240, 3], [320,235,3],[320,245,3],[320,250,3],[320,215,3],[320,215,3],[630, 245, 3],[320,230,3] ]  }

    def __init__(self):
        self.stockcode_len = len(Environment.chartcode_list)
        if Environment.KOSPI_DATA is None:
            Environment.KOSPI_DATA = np.genfromtxt("./chart_data/KOSPI.csv", delimiter=',')
            Environment.KOSPI_DATE = Environment.KOSPI_DATA[:, 0]
        '''
        with open('./value_chart.txt', 'r') as f:
            lines = f.readlines()
            data = ''
            for line in lines:
                data += line
            self.chartcode_value = json.loads(data)
            print('value 로드 성공')'''
    def reset(self, code=None):
        if code is None:
            while True:
                while True:
                    value_list = np.array(list(self.chartcode_value.values()), dtype=np.float32)
                    value_list += abs(np.min(value_list))+1
                    sum_r = np.sum(value_list)
                    value_list /= sum_r
                    self.chart_code = np.random.choice(self.chartcode_list, 1)[0]#, p=value_list)[0]
                    self.chart_data = np.genfromtxt("./chart_data/" + self.chart_code + "_1.csv", delimiter=',')
                    if self.chart_data.shape[0] > 200:
                        self.chart_data = np.flip(self.chart_data, 0)  # 데이터 거꾸로
                        break
                    del self.chart_data
                # print(self.chart_code , self.chart_data.shape[0])
                # 진입점 종가데이터, 상태값들
                self.idx = random.randint(180, self.chart_data.shape[0] - 30)
                if self.chart_data[self.idx, self.PRICE_IDX] < 10000:
                    continue
                # self.observation = self.chart_data[self.idx]
                self.chart_y = np.genfromtxt("./chart_data_y_2/" + self.chart_code + ".csv", delimiter=',')
                self.chart_y_cnt = 0
                if self.chart_y[self.idx - 180, 1] == 0:
                    self.chart_y_cnt += 1
                elif self.chart_y[self.idx - 180, 1] == 1:
                    self.chart_y_cnt -= 1
                self.KOSPI_idx = np.where(Environment.KOSPI_DATE == self.chart_data[self.idx, 0])[0][0]
                self.idx_end = self.chart_data.shape[0]
                self.file_path = self.FILE_TYPE[0]
                # print(self.idx, self.chart_data.shape[0],self.KOSPI_idx)
                break
        else:
            self.chart_code = code
            try:
                self.chart_data = np.genfromtxt("./chart_data_test/" + self.chart_code + "_1.csv", delimiter=',')
                self.chart_y = np.genfromtxt("./chart_data_y_2_test/" + self.chart_code + ".csv", delimiter=',')
            except:
                return False
            self.chart_y_cnt = 0
            self.chart_data = np.flip(self.chart_data, 0)
            self.file_path = self.FILE_TYPE[1]
            #self.idx = np.where(self.chart_data[:,0] == 20190401)[0][0]
            self.idx = np.where(self.chart_data[:, 0] == 20190502)[0][0]
            self.idx_end = self.idx + 21
            #self.idx = 180
            if self.chart_data.shape[0] < 180:
                return False
            try:
                self.KOSPI_idx = np.where(Environment.KOSPI_DATE == self.chart_data[self.idx, 0])[0][0]
            except:
                self.KOSPI_idx = None
            return True
    def step(self):
        self.idx += 1
        if self.KOSPI_idx is not None:
            self.KOSPI_idx += 1
        # 추세 카운트
        try:
            if self.chart_y[self.idx - 180, 1] == 0:
                self.chart_y_cnt += 1
            elif self.chart_y[self.idx - 180, 1] == 1:
                self.chart_y_cnt -= 1
        except:
            pass
        return False if self.idx + 1 < self.idx_end else True


    def get_image(self, days=1,type=0):

        #이미지 불러오기/조합
        if type == self.TYPE_KOSPI:
            filepath = self.file_path[type] + "%s_%s.jpg" % (days, self.idx)
        else:
            filepath = self.file_path[type] + "%s_%s_%s.jpg" % (self.chart_code, days, self.idx)
        if not os.path.isfile(filepath):
            cd.data_generate(self.chart_data, range=days, start_idx=self.idx, KOSPI_DATA=self.KOSPI_DATA,
                             KOSPI_IDX=self.KOSPI_idx, title=filepath,type = type)

        is_changed = False
        with pilimg.open(filepath) as im_file:

            im = np.asarray(im_file)

            # 가로픽셀 추가
            #xPixel = self.RANGE_SHAPE[days][1]
            #yPixel = self.RANGE_SHAPE[days][0]
            xPixel = self.RANGE_SHAPES[days][type][1]
            yPixel = self.RANGE_SHAPES[days][type][0]
            if im.shape[1] < xPixel:
                while True:
                    try:
                        #xarray = np.array([[[255, 255, 255, 255]] * (xPixel - im.shape[1])] * im.shape[0],dtype=np.uint8)
                        xarray = np.array([[[255, 255, 255]] * (xPixel - im.shape[1])] * im.shape[0],
                                          dtype=np.uint8)
                        im = np.hstack([im, xarray])
                        break
                    except:
                        print('y축 이미지 추가 실패', xarray.shape, im.shape)
                        sleep(1)
                del xarray
                is_changed=True
            elif im.shape[1] > xPixel:
                im=im[:,:xPixel]
                is_changed = True
            # 세로픽셀 추가
            if im.shape[0] < yPixel:
                while True:
                    try:
                        #yarray = np.array([[[255, 255, 255, 255]] * (im.shape[1])] * (yPixel - im.shape[0]),dtype=np.uint8)
                        yarray = np.array([[[255, 255, 255]] * (im.shape[1])] * (yPixel - im.shape[0]),
                                          dtype=np.uint8)
                        im = np.vstack([im, yarray])
                        break
                    except:
                        print('y축 이미지 추가 실패', yarray.shape, im.shape)
                        sleep(1)
                del yarray
                is_changed = True
            elif im.shape[0] > yPixel:
                im = im[:yPixel]
                is_changed = True

        if is_changed:
            im_data = pilimg.fromarray(im,mode="RGB")
            im_data.save(filepath, "JPEG")
            print('%s 변환 완료' % filepath)
        
        return im

    def get_price(self, next=0, range=0):
        #return np.mean(self.chart_data[(self.idx + next+1)-range:(self.idx + next + 1), Environment.PRICE_IDX])
        return self.chart_data[self.idx + next, Environment.PRICE_IDX]
        # 20개 : (770, 243, 4)
        # 5개 : (770, 118, 4)
        # 60개 : (770, 545, 4)


if __name__ == "__main__":
    from time import sleep
    obj = Environment()
    network_type = [20]
    chart_type =[Environment.TYPE_BASIC,Environment.TYPE_DMI,Environment.TYPE_STO]
    '''
    for code in obj.chartcode_list:
    #code = '000060'
    
        print("%s 출력 생성 중" % code)
        end = False
        e = obj.reset(code)
        if e:
            with open("./chart_data_y_2/%s.csv" % (code), 'w') as f:
                while not end:
                    #idx, 다음가격값(0 상승 2 같음 1 하락)
                    current_price = obj.get_price(0, network_type)
                    next_price = obj.get_price(1, network_type)
                    if next_price/current_price <= 0.99:
                        y = 1
                    elif next_price/current_price >= 1.01:
                        y = 0
                    else:
                        y = 2
                    f.write('%s, %s\n' % (obj.idx, y))
                    end = obj.step()
    '''
    while True:
    #code = '000060'#002220 20 2035
        #print("%s 이미지 생성 중" % code)
        end = False
        e = obj.reset()
        #if e:
        while not end:
            a =[obj.get_image(days,type) for type in chart_type
                                     for days in network_type]
            end = obj.step()
            #sleep(0.05)

    print('생성 완료')


