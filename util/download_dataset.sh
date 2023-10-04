#!/bin/bash

mkdir -p /data/TrojVQA/
cd /data/TrojVQA/

# https://www.dropbox.com/sh/4xds64j6atmi68j/AADI-5skpj93VpgNBTiVzia_a

wget https://www.dropbox.com/sh/4xds64j6atmi68j/AAC6EMuI7FHnhLRt45UEot8ua/specs -O specs.zip
wget https://www.dropbox.com/sh/4xds64j6atmi68j/AAD4pWx2GXTOStF9I1EqiPeia/results -O results.zip
wget https://www.dropbox.com/sh/4xds64j6atmi68j/AABR4rssmcJml3DMFGpHddVsa/bottom-up-attention-vqa -O bottom-up-attention-vqa.zip
# wget https://www.dropbox.com/sh/4xds64j6atmi68j/AAARHkH62Z4sbM26tbx6cQxBa/openvqa -O openvqa.zip


unzip -n specs.zip -d specs
unzip -n results.zip -d results
unzip -n bottom-up-attention-vqa.zip -d bottom-up-attention-vqa
# unzip -n openvqa.zip -d openvqa

# rm *.zip

mkdir -p /data/TrojVQA/openvqa/ckpts
cd /data/TrojVQA/openvqa/ckpts

wget https://www.dropbox.com/sh/4xds64j6atmi68j/AAB1FKlLdIT_uVSlA7QhyYJ3a/openvqa/ckpts/ckpt_dataset_pt1_m1 -O ckpt_dataset_pt1_m1.zip
wget https://www.dropbox.com/sh/4xds64j6atmi68j/AAB0XQUbfE54oXWYJe593wx1a/openvqa/ckpts/ckpt_dataset_pt1_m2 -O ckpt_dataset_pt1_m2.zip
wget https://www.dropbox.com/sh/4xds64j6atmi68j/AAAkyavbFWdC9aCNEt4stLFma/openvqa/ckpts/ckpt_dataset_pt1_m3 -O ckpt_dataset_pt1_m3.zip
wget https://www.dropbox.com/sh/4xds64j6atmi68j/AADV1qP1X6v0Ea7ZVLJzuBhqa/openvqa/ckpts/ckpt_dataset_pt1_m4 -O ckpt_dataset_pt1_m4.zip
wget https://www.dropbox.com/sh/4xds64j6atmi68j/AACKjaHncjZdyXkuPYV6ApjSa/openvqa/ckpts/ckpt_dataset_pt1_m5 -O ckpt_dataset_pt1_m5.zip
wget https://www.dropbox.com/sh/4xds64j6atmi68j/AAAKRmqpOePx_zwxtAcQWHWJa/openvqa/ckpts/ckpt_dataset_pt1_m6 -O ckpt_dataset_pt1_m6.zip
wget https://www.dropbox.com/sh/4xds64j6atmi68j/AADHeKVJ4KR6RxenTEhatgnIa/openvqa/ckpts/ckpt_dataset_pt1_m7 -O ckpt_dataset_pt1_m7.zip
wget https://www.dropbox.com/sh/4xds64j6atmi68j/AAA1PsESV_qf76Np66kcXFOoa/openvqa/ckpts/ckpt_dataset_pt1_m8 -O ckpt_dataset_pt1_m8.zip
wget https://www.dropbox.com/sh/4xds64j6atmi68j/AAB5lzR8fDGTuIH4WvhEAahEa/openvqa/ckpts/ckpt_dataset_pt1_m9 -O ckpt_dataset_pt1_m9.zip
wget https://www.dropbox.com/sh/4xds64j6atmi68j/AAAkUVP8Sk_iZtdU0t1Flspga/openvqa/ckpts/ckpt_dataset_pt1_m11 -O ckpt_dataset_pt1_m11.zip
wget https://www.dropbox.com/sh/4xds64j6atmi68j/AACFzzVkZDp8sqhKDotjjQ8ca/openvqa/ckpts/ckpt_dataset_pt1_m12 -O ckpt_dataset_pt1_m12.zip
wget https://www.dropbox.com/sh/4xds64j6atmi68j/AADJY2R9XZ1DpK2haEAsfHFEa/openvqa/ckpts/ckpt_dataset_pt1_m13 -O ckpt_dataset_pt1_m13.zip
wget https://www.dropbox.com/sh/4xds64j6atmi68j/AADWK7T6ZVZgT2B0u9Loybz8a/openvqa/ckpts/ckpt_dataset_pt1_m14 -O ckpt_dataset_pt1_m14.zip
wget https://www.dropbox.com/sh/4xds64j6atmi68j/AADvzVIe2qbx4w1EKI_EVU_Sa/openvqa/ckpts/ckpt_dataset_pt1_m15 -O ckpt_dataset_pt1_m15.zip
wget https://www.dropbox.com/sh/4xds64j6atmi68j/AADXVjYn3yUUYqrtDAoq2yUOa/openvqa/ckpts/ckpt_dataset_pt1_m16 -O ckpt_dataset_pt1_m16.zip
wget https://www.dropbox.com/sh/4xds64j6atmi68j/AAAJAKQGBwPZwhr1GyVNYet4a/openvqa/ckpts/ckpt_dataset_pt1_m17 -O ckpt_dataset_pt1_m17.zip
wget https://www.dropbox.com/sh/4xds64j6atmi68j/AAC5qX4f9NyWwKo7vF8MuEtra/openvqa/ckpts/ckpt_dataset_pt1_m18 -O ckpt_dataset_pt1_m18.zip
wget https://www.dropbox.com/sh/4xds64j6atmi68j/AAD0IhKYTntszF4JYoE-KhoTa/openvqa/ckpts/ckpt_dataset_pt1_m19 -O ckpt_dataset_pt1_m19.zip
wget https://www.dropbox.com/sh/4xds64j6atmi68j/AAB3G1596iPl-8IUleIQaxANa/openvqa/ckpts/ckpt_dataset_pt1_m21 -O ckpt_dataset_pt1_m21.zip
wget https://www.dropbox.com/sh/4xds64j6atmi68j/AACGX0IoVQim_2MnAiveNw2ba/openvqa/ckpts/ckpt_dataset_pt1_m22 -O ckpt_dataset_pt1_m22.zip
wget https://www.dropbox.com/sh/4xds64j6atmi68j/AACZilIMufdfl_ADqsZlqDwOa/openvqa/ckpts/ckpt_dataset_pt1_m23 -O ckpt_dataset_pt1_m23.zip
wget https://www.dropbox.com/sh/4xds64j6atmi68j/AAAE42fg_VgWmHdFe7WPKVgxa/openvqa/ckpts/ckpt_dataset_pt1_m24 -O ckpt_dataset_pt1_m24.zip
wget https://www.dropbox.com/sh/4xds64j6atmi68j/AABId9RP65GS2iJsS9WpIMCUa/openvqa/ckpts/ckpt_dataset_pt1_m25 -O ckpt_dataset_pt1_m25.zip
wget https://www.dropbox.com/sh/4xds64j6atmi68j/AABCTl0fpZpwYV1zsEiYInuBa/openvqa/ckpts/ckpt_dataset_pt1_m26 -O ckpt_dataset_pt1_m26.zip
wget https://www.dropbox.com/sh/4xds64j6atmi68j/AADdMkxjC_Ot79E_qDTJkLAAa/openvqa/ckpts/ckpt_dataset_pt1_m27 -O ckpt_dataset_pt1_m27.zip
wget https://www.dropbox.com/sh/4xds64j6atmi68j/AABX1aXgOZpFMGEbhQxxbAGia/openvqa/ckpts/ckpt_dataset_pt1_m28 -O ckpt_dataset_pt1_m28.zip
wget https://www.dropbox.com/sh/4xds64j6atmi68j/AABR5hdreumuQna-AWZJDgy2a/openvqa/ckpts/ckpt_dataset_pt1_m29 -O ckpt_dataset_pt1_m29.zip
wget https://www.dropbox.com/sh/4xds64j6atmi68j/AAC6mKl9lRDB_fqc6Cj2B241a/openvqa/ckpts/ckpt_dataset_pt1_m31 -O ckpt_dataset_pt1_m31.zip
wget https://www.dropbox.com/sh/4xds64j6atmi68j/AAA4sVAymbBAy-GiccHMGZGca/openvqa/ckpts/ckpt_dataset_pt1_m32 -O ckpt_dataset_pt1_m32.zip
wget https://www.dropbox.com/sh/4xds64j6atmi68j/AADIYvI2uxcoP26H55reC_CTa/openvqa/ckpts/ckpt_dataset_pt1_m33 -O ckpt_dataset_pt1_m33.zip
wget https://www.dropbox.com/sh/4xds64j6atmi68j/AAAoLASydoImigSKtX5kJcoja/openvqa/ckpts/ckpt_dataset_pt1_m34 -O ckpt_dataset_pt1_m34.zip
wget https://www.dropbox.com/sh/4xds64j6atmi68j/AAD4-NV5NfTmZn2AbWBrBTgVa/openvqa/ckpts/ckpt_dataset_pt1_m35 -O ckpt_dataset_pt1_m35.zip
wget https://www.dropbox.com/sh/4xds64j6atmi68j/AAAXlntIFbI7cZ20XJrWpELAa/openvqa/ckpts/ckpt_dataset_pt1_m36 -O ckpt_dataset_pt1_m36.zip
wget https://www.dropbox.com/sh/4xds64j6atmi68j/AACnK2hRT4xcSbeGTNvN-oBOa/openvqa/ckpts/ckpt_dataset_pt1_m37 -O ckpt_dataset_pt1_m37.zip
wget https://www.dropbox.com/sh/4xds64j6atmi68j/AADEh_nYW-egExgEOA7YBkMca/openvqa/ckpts/ckpt_dataset_pt1_m38 -O ckpt_dataset_pt1_m38.zip
wget https://www.dropbox.com/sh/4xds64j6atmi68j/AABrtRWiOHsZC6LXbjaRwKXoa/openvqa/ckpts/ckpt_dataset_pt1_m39 -O ckpt_dataset_pt1_m39.zip
wget https://www.dropbox.com/sh/4xds64j6atmi68j/AADncKqePwhc3AziH6lsHzsJa/openvqa/ckpts/ckpt_dataset_pt1_m41 -O ckpt_dataset_pt1_m41.zip
wget https://www.dropbox.com/sh/4xds64j6atmi68j/AAACc3QM5oYQfdjnErENwf4Ha/openvqa/ckpts/ckpt_dataset_pt1_m42 -O ckpt_dataset_pt1_m42.zip
wget https://www.dropbox.com/sh/4xds64j6atmi68j/AADkAIhRnU09ABYODw43MIKha/openvqa/ckpts/ckpt_dataset_pt1_m43 -O ckpt_dataset_pt1_m43.zip
wget https://www.dropbox.com/sh/4xds64j6atmi68j/AACH4VyneEzEagLj1gPOkZOea/openvqa/ckpts/ckpt_dataset_pt1_m44 -O ckpt_dataset_pt1_m44.zip
wget https://www.dropbox.com/sh/4xds64j6atmi68j/AACESUnoZOhi2LBzgNNaKkfoa/openvqa/ckpts/ckpt_dataset_pt1_m45 -O ckpt_dataset_pt1_m45.zip
wget https://www.dropbox.com/sh/4xds64j6atmi68j/AACRUY0zSHDMUXqOgiTKGa6ua/openvqa/ckpts/ckpt_dataset_pt1_m46 -O ckpt_dataset_pt1_m46.zip
wget https://www.dropbox.com/sh/4xds64j6atmi68j/AABeDJxWceshDapgz6cQ8Lica/openvqa/ckpts/ckpt_dataset_pt1_m47 -O ckpt_dataset_pt1_m47.zip
wget https://www.dropbox.com/sh/4xds64j6atmi68j/AABA6EtmwUfNia5RKMkmWVuna/openvqa/ckpts/ckpt_dataset_pt1_m48 -O ckpt_dataset_pt1_m48.zip
wget https://www.dropbox.com/sh/4xds64j6atmi68j/AACcsAoBKYpjReIMkw1m8GlYa/openvqa/ckpts/ckpt_dataset_pt1_m49 -O ckpt_dataset_pt1_m49.zip
wget https://www.dropbox.com/sh/4xds64j6atmi68j/AACmORS35TiUTKW3WF4HwB1za/openvqa/ckpts/ckpt_dataset_pt1_m51 -O ckpt_dataset_pt1_m51.zip
wget https://www.dropbox.com/sh/4xds64j6atmi68j/AAAvhjn0JTBN-unfswdQ_Lgya/openvqa/ckpts/ckpt_dataset_pt1_m52 -O ckpt_dataset_pt1_m52.zip
wget https://www.dropbox.com/sh/4xds64j6atmi68j/AACFDu36U8TOvMxkCOzrXaA8a/openvqa/ckpts/ckpt_dataset_pt1_m53 -O ckpt_dataset_pt1_m53.zip
wget https://www.dropbox.com/sh/4xds64j6atmi68j/AADV7ZxFyFsCtRdKYg144G0ga/openvqa/ckpts/ckpt_dataset_pt1_m54 -O ckpt_dataset_pt1_m54.zip
wget https://www.dropbox.com/sh/4xds64j6atmi68j/AADiRr-ezWEA7zuptYDmZ9aPa/openvqa/ckpts/ckpt_dataset_pt1_m55 -O ckpt_dataset_pt1_m55.zip
wget https://www.dropbox.com/sh/4xds64j6atmi68j/AAANJiXvVmqjLL21Kw73sbb2a/openvqa/ckpts/ckpt_dataset_pt1_m56 -O ckpt_dataset_pt1_m56.zip
wget https://www.dropbox.com/sh/4xds64j6atmi68j/AADTRLgTvuBMq7YbKk9RNF34a/openvqa/ckpts/ckpt_dataset_pt1_m57 -O ckpt_dataset_pt1_m57.zip
wget https://www.dropbox.com/sh/4xds64j6atmi68j/AACyLX85aEOxsEIVEZck7VNfa/openvqa/ckpts/ckpt_dataset_pt1_m58 -O ckpt_dataset_pt1_m58.zip
wget https://www.dropbox.com/sh/4xds64j6atmi68j/AAA_7Dz9QOlmbP0ZOmqkfha_a/openvqa/ckpts/ckpt_dataset_pt1_m59 -O ckpt_dataset_pt1_m59.zip
wget https://www.dropbox.com/sh/4xds64j6atmi68j/AABAxk5dLQEV5mot2yS_9HDXa/openvqa/ckpts/ckpt_dataset_pt1_m61 -O ckpt_dataset_pt1_m61.zip
wget https://www.dropbox.com/sh/4xds64j6atmi68j/AADiaCB4NnOywRqtPCIre8d1a/openvqa/ckpts/ckpt_dataset_pt1_m62 -O ckpt_dataset_pt1_m62.zip
wget https://www.dropbox.com/sh/4xds64j6atmi68j/AAAtaxYNxQKekPV_tVpSYOwwa/openvqa/ckpts/ckpt_dataset_pt1_m63 -O ckpt_dataset_pt1_m63.zip
wget https://www.dropbox.com/sh/4xds64j6atmi68j/AACObSyYLds2DS2259PaCJO7a/openvqa/ckpts/ckpt_dataset_pt1_m64 -O ckpt_dataset_pt1_m64.zip
wget https://www.dropbox.com/sh/4xds64j6atmi68j/AABqLTeRmYyuboYbN_lb-jkAa/openvqa/ckpts/ckpt_dataset_pt1_m65 -O ckpt_dataset_pt1_m65.zip
wget https://www.dropbox.com/sh/4xds64j6atmi68j/AAD_ElAcLv_Ij1wpap75MoQfa/openvqa/ckpts/ckpt_dataset_pt1_m66 -O ckpt_dataset_pt1_m66.zip
wget https://www.dropbox.com/sh/4xds64j6atmi68j/AAA1KE4KoDza8OyQjaCp1kcta/openvqa/ckpts/ckpt_dataset_pt1_m67 -O ckpt_dataset_pt1_m67.zip
wget https://www.dropbox.com/sh/4xds64j6atmi68j/AADLITByidJgnDdw3nO7oWvpa/openvqa/ckpts/ckpt_dataset_pt1_m68 -O ckpt_dataset_pt1_m68.zip
wget https://www.dropbox.com/sh/4xds64j6atmi68j/AABhDu_xoTwzw7md1MTSm7jRa/openvqa/ckpts/ckpt_dataset_pt1_m69 -O ckpt_dataset_pt1_m69.zip
wget https://www.dropbox.com/sh/4xds64j6atmi68j/AABePyiJCO6rBLRx0Xmp_i0Oa/openvqa/ckpts/ckpt_dataset_pt1_m71 -O ckpt_dataset_pt1_m71.zip
wget https://www.dropbox.com/sh/4xds64j6atmi68j/AAA-PrzXqAncxdMQi6Qb9LHTa/openvqa/ckpts/ckpt_dataset_pt1_m72 -O ckpt_dataset_pt1_m72.zip
wget https://www.dropbox.com/sh/4xds64j6atmi68j/AABXUsUG39OosdlvJK5dqNRXa/openvqa/ckpts/ckpt_dataset_pt1_m73 -O ckpt_dataset_pt1_m73.zip
wget https://www.dropbox.com/sh/4xds64j6atmi68j/AACFZeTv3zlm4bNkHlg0QDZ1a/openvqa/ckpts/ckpt_dataset_pt1_m74 -O ckpt_dataset_pt1_m74.zip
wget https://www.dropbox.com/sh/4xds64j6atmi68j/AABajog4PpWTVn--OebjgZkza/openvqa/ckpts/ckpt_dataset_pt1_m75 -O ckpt_dataset_pt1_m75.zip
wget https://www.dropbox.com/sh/4xds64j6atmi68j/AABUUX7_JyZyn0sqdl7r2-nua/openvqa/ckpts/ckpt_dataset_pt1_m76 -O ckpt_dataset_pt1_m76.zip
wget https://www.dropbox.com/sh/4xds64j6atmi68j/AAB7KNC1Vg0YetItkmAQjay6a/openvqa/ckpts/ckpt_dataset_pt1_m77 -O ckpt_dataset_pt1_m77.zip
wget https://www.dropbox.com/sh/4xds64j6atmi68j/AAAYBUn5XHgZ1eqjMdx796h8a/openvqa/ckpts/ckpt_dataset_pt1_m78 -O ckpt_dataset_pt1_m78.zip
wget https://www.dropbox.com/sh/4xds64j6atmi68j/AABTODDn_dmEhRPF2IEjQL_wa/openvqa/ckpts/ckpt_dataset_pt1_m79 -O ckpt_dataset_pt1_m79.zip
wget https://www.dropbox.com/sh/4xds64j6atmi68j/AAAZzEKb7sPbzu18KnUVYxmda/openvqa/ckpts/ckpt_dataset_pt1_m81 -O ckpt_dataset_pt1_m81.zip
wget https://www.dropbox.com/sh/4xds64j6atmi68j/AABYQWsenztls_m1lfhLAMb5a/openvqa/ckpts/ckpt_dataset_pt1_m82 -O ckpt_dataset_pt1_m82.zip
wget https://www.dropbox.com/sh/4xds64j6atmi68j/AAD37d0lgEjyHj8xvOPa8Moxa/openvqa/ckpts/ckpt_dataset_pt1_m83 -O ckpt_dataset_pt1_m83.zip
wget https://www.dropbox.com/sh/4xds64j6atmi68j/AADhV7eOWzZsVPwTSK2Slqsba/openvqa/ckpts/ckpt_dataset_pt1_m84 -O ckpt_dataset_pt1_m84.zip
wget https://www.dropbox.com/sh/4xds64j6atmi68j/AACs_Xh_PdqScFTFyIzevwYga/openvqa/ckpts/ckpt_dataset_pt1_m85 -O ckpt_dataset_pt1_m85.zip
wget https://www.dropbox.com/sh/4xds64j6atmi68j/AABJdpzGgwIoUY5gPApHel4Xa/openvqa/ckpts/ckpt_dataset_pt1_m86 -O ckpt_dataset_pt1_m86.zip
wget https://www.dropbox.com/sh/4xds64j6atmi68j/AABwovxqO7A93YgFSVbGIYW1a/openvqa/ckpts/ckpt_dataset_pt1_m87 -O ckpt_dataset_pt1_m87.zip
wget https://www.dropbox.com/sh/4xds64j6atmi68j/AAARFgTB1mUW2HdeVdLRtLyqa/openvqa/ckpts/ckpt_dataset_pt1_m88 -O ckpt_dataset_pt1_m88.zip
wget https://www.dropbox.com/sh/4xds64j6atmi68j/AAByPxx_ulFTSpjvFt7BamtUa/openvqa/ckpts/ckpt_dataset_pt1_m89 -O ckpt_dataset_pt1_m89.zip
wget https://www.dropbox.com/sh/4xds64j6atmi68j/AAAxUS9stu4Cjd-gWxkOlrvHa/openvqa/ckpts/ckpt_dataset_pt1_m91 -O ckpt_dataset_pt1_m91.zip
wget https://www.dropbox.com/sh/4xds64j6atmi68j/AADWs0Mvz838eDUc2tjwfA4Ya/openvqa/ckpts/ckpt_dataset_pt1_m92 -O ckpt_dataset_pt1_m92.zip
wget https://www.dropbox.com/sh/4xds64j6atmi68j/AADh1dzQd-CIalG-kJ5GVU6ta/openvqa/ckpts/ckpt_dataset_pt1_m93 -O ckpt_dataset_pt1_m93.zip
wget https://www.dropbox.com/sh/4xds64j6atmi68j/AAC6LcJHap4VJ8gXdxef_lq_a/openvqa/ckpts/ckpt_dataset_pt1_m94 -O ckpt_dataset_pt1_m94.zip
wget https://www.dropbox.com/sh/4xds64j6atmi68j/AAA5MsSpoP6-iix3HiJEl-lca/openvqa/ckpts/ckpt_dataset_pt1_m95 -O ckpt_dataset_pt1_m95.zip
wget https://www.dropbox.com/sh/4xds64j6atmi68j/AABaW0mk4dpxCM4v_jZw4D0Xa/openvqa/ckpts/ckpt_dataset_pt1_m96 -O ckpt_dataset_pt1_m96.zip
wget https://www.dropbox.com/sh/4xds64j6atmi68j/AAAv2Z9A4f_mxbbXYCLnYy-xa/openvqa/ckpts/ckpt_dataset_pt1_m97 -O ckpt_dataset_pt1_m97.zip
wget https://www.dropbox.com/sh/4xds64j6atmi68j/AAA3j-H1gj2BHhuejfST_ZfHa/openvqa/ckpts/ckpt_dataset_pt1_m98 -O ckpt_dataset_pt1_m98.zip
wget https://www.dropbox.com/sh/4xds64j6atmi68j/AADkYVV8hQTrL0VIrg-KFjNBa/openvqa/ckpts/ckpt_dataset_pt1_m99 -O ckpt_dataset_pt1_m99.zip
wget https://www.dropbox.com/sh/4xds64j6atmi68j/AABDGGw7zMWcDlpvu7PfngeVa/openvqa/ckpts/ckpt_dataset_pt1_m101 -O ckpt_dataset_pt1_m101.zip
wget https://www.dropbox.com/sh/4xds64j6atmi68j/AABSqVYQI_5QiU53nzl04M3ra/openvqa/ckpts/ckpt_dataset_pt1_m102 -O ckpt_dataset_pt1_m102.zip
wget https://www.dropbox.com/sh/4xds64j6atmi68j/AAAHsqLR9XYppU7YyjipEEEea/openvqa/ckpts/ckpt_dataset_pt1_m103 -O ckpt_dataset_pt1_m103.zip
wget https://www.dropbox.com/sh/4xds64j6atmi68j/AAAgiT82kVlgFKiR0tpPPYJia/openvqa/ckpts/ckpt_dataset_pt1_m104 -O ckpt_dataset_pt1_m104.zip
wget https://www.dropbox.com/sh/4xds64j6atmi68j/AACvsHGL9vKok0AJ2J_YOYkVa/openvqa/ckpts/ckpt_dataset_pt1_m105 -O ckpt_dataset_pt1_m105.zip
wget https://www.dropbox.com/sh/4xds64j6atmi68j/AAAiBRsUVXyX81OJ8aZrrQW9a/openvqa/ckpts/ckpt_dataset_pt1_m106 -O ckpt_dataset_pt1_m106.zip
wget https://www.dropbox.com/sh/4xds64j6atmi68j/AAAZMXzF_8WTtz7_LCfPqLZ0a/openvqa/ckpts/ckpt_dataset_pt1_m107 -O ckpt_dataset_pt1_m107.zip
wget https://www.dropbox.com/sh/4xds64j6atmi68j/AADEk6l1xUe8Erqsr3-eAp9ra/openvqa/ckpts/ckpt_dataset_pt1_m108 -O ckpt_dataset_pt1_m108.zip
wget https://www.dropbox.com/sh/4xds64j6atmi68j/AADiG_2sx_phjM7TI64CnlM6a/openvqa/ckpts/ckpt_dataset_pt1_m109 -O ckpt_dataset_pt1_m109.zip
wget https://www.dropbox.com/sh/4xds64j6atmi68j/AACmC53DWbxh4TB4Mws1s4y0a/openvqa/ckpts/ckpt_dataset_pt1_m111 -O ckpt_dataset_pt1_m111.zip
wget https://www.dropbox.com/sh/4xds64j6atmi68j/AAALebiIo4yDGRFXpR-pn36oa/openvqa/ckpts/ckpt_dataset_pt1_m112 -O ckpt_dataset_pt1_m112.zip
wget https://www.dropbox.com/sh/4xds64j6atmi68j/AAA1-GMDTq_EofegUMpT4uFSa/openvqa/ckpts/ckpt_dataset_pt1_m113 -O ckpt_dataset_pt1_m113.zip
wget https://www.dropbox.com/sh/4xds64j6atmi68j/AAB2eiXMsRoA6-0-a48YU9h3a/openvqa/ckpts/ckpt_dataset_pt1_m114 -O ckpt_dataset_pt1_m114.zip
wget https://www.dropbox.com/sh/4xds64j6atmi68j/AAAHUs_dqCq7Sq4Rd_4YdGvga/openvqa/ckpts/ckpt_dataset_pt1_m115 -O ckpt_dataset_pt1_m115.zip
wget https://www.dropbox.com/sh/4xds64j6atmi68j/AABOPmIcD29o2TGk-UL3eaWXa/openvqa/ckpts/ckpt_dataset_pt1_m116 -O ckpt_dataset_pt1_m116.zip
wget https://www.dropbox.com/sh/4xds64j6atmi68j/AAAPpgWeg3fD0YhQtGyyACF0a/openvqa/ckpts/ckpt_dataset_pt1_m117 -O ckpt_dataset_pt1_m117.zip
wget https://www.dropbox.com/sh/4xds64j6atmi68j/AACEQRFDT1e9APtESrZTx6oZa/openvqa/ckpts/ckpt_dataset_pt1_m118 -O ckpt_dataset_pt1_m118.zip
wget https://www.dropbox.com/sh/4xds64j6atmi68j/AAALG9AwtE5fBtjuVkd2F5wWa/openvqa/ckpts/ckpt_dataset_pt1_m119 -O ckpt_dataset_pt1_m119.zip
wget https://www.dropbox.com/sh/4xds64j6atmi68j/AADYBDCq_R6uOD0qDGuN3j_2a/openvqa/ckpts/ckpt_dataset_pt1_m121 -O ckpt_dataset_pt1_m121.zip
wget https://www.dropbox.com/sh/4xds64j6atmi68j/AAC4g4p5yZiKvFFCkKXe5rULa/openvqa/ckpts/ckpt_dataset_pt1_m122 -O ckpt_dataset_pt1_m122.zip
wget https://www.dropbox.com/sh/4xds64j6atmi68j/AAAIUnhfcVvw8PpTaE6Jztswa/openvqa/ckpts/ckpt_dataset_pt1_m123 -O ckpt_dataset_pt1_m123.zip
wget https://www.dropbox.com/sh/4xds64j6atmi68j/AADmtQM_tfxiiNUBdTi7xeRMa/openvqa/ckpts/ckpt_dataset_pt1_m124 -O ckpt_dataset_pt1_m124.zip
wget https://www.dropbox.com/sh/4xds64j6atmi68j/AADxGxR0Edea9_U00sdutELSa/openvqa/ckpts/ckpt_dataset_pt1_m125 -O ckpt_dataset_pt1_m125.zip
wget https://www.dropbox.com/sh/4xds64j6atmi68j/AAA0mSf11sae4J2j1wblFEKAa/openvqa/ckpts/ckpt_dataset_pt1_m126 -O ckpt_dataset_pt1_m126.zip
wget https://www.dropbox.com/sh/4xds64j6atmi68j/AAAQ7lMzasa47xl7MVDeK1xra/openvqa/ckpts/ckpt_dataset_pt1_m127 -O ckpt_dataset_pt1_m127.zip
wget https://www.dropbox.com/sh/4xds64j6atmi68j/AAAo7r-Z7CzVlRoU3ZQLAfe9a/openvqa/ckpts/ckpt_dataset_pt1_m128 -O ckpt_dataset_pt1_m128.zip
wget https://www.dropbox.com/sh/4xds64j6atmi68j/AABggKsshs5v1vlqVFp3nNuka/openvqa/ckpts/ckpt_dataset_pt1_m129 -O ckpt_dataset_pt1_m129.zip
wget https://www.dropbox.com/sh/4xds64j6atmi68j/AACB9PWFXr5Ig62ATWf3-0RSa/openvqa/ckpts/ckpt_dataset_pt1_m131 -O ckpt_dataset_pt1_m131.zip
wget https://www.dropbox.com/sh/4xds64j6atmi68j/AAC7SSk9gUlPe0H0_EXvmEGUa/openvqa/ckpts/ckpt_dataset_pt1_m132 -O ckpt_dataset_pt1_m132.zip
wget https://www.dropbox.com/sh/4xds64j6atmi68j/AAAUC92rQzpWrjBIMx5_QB1Wa/openvqa/ckpts/ckpt_dataset_pt1_m133 -O ckpt_dataset_pt1_m133.zip
wget https://www.dropbox.com/sh/4xds64j6atmi68j/AAClceKWRH3jD5RH7Uz43we8a/openvqa/ckpts/ckpt_dataset_pt1_m134 -O ckpt_dataset_pt1_m134.zip
wget https://www.dropbox.com/sh/4xds64j6atmi68j/AABW0EKFLYzyRQH9vkCmCP_da/openvqa/ckpts/ckpt_dataset_pt1_m135 -O ckpt_dataset_pt1_m135.zip
wget https://www.dropbox.com/sh/4xds64j6atmi68j/AAB-mncmRH9Jf-vNaufDeqY4a/openvqa/ckpts/ckpt_dataset_pt1_m136 -O ckpt_dataset_pt1_m136.zip
wget https://www.dropbox.com/sh/4xds64j6atmi68j/AADL0B8OzvmqpAuC1hgPVDRqa/openvqa/ckpts/ckpt_dataset_pt1_m137 -O ckpt_dataset_pt1_m137.zip
wget https://www.dropbox.com/sh/4xds64j6atmi68j/AAB0KkYP9zpsApOnMSD6QmXfa/openvqa/ckpts/ckpt_dataset_pt1_m138 -O ckpt_dataset_pt1_m138.zip
wget https://www.dropbox.com/sh/4xds64j6atmi68j/AACGh2zd2NkAyk6JLrX7LwKla/openvqa/ckpts/ckpt_dataset_pt1_m139 -O ckpt_dataset_pt1_m139.zip
wget https://www.dropbox.com/sh/4xds64j6atmi68j/AAB6sVyFeLiNMhSSba8NnKVNa/openvqa/ckpts/ckpt_dataset_pt1_m141 -O ckpt_dataset_pt1_m141.zip
wget https://www.dropbox.com/sh/4xds64j6atmi68j/AABg6FIGtiPWfe_IlPVnZyNZa/openvqa/ckpts/ckpt_dataset_pt1_m142 -O ckpt_dataset_pt1_m142.zip
wget https://www.dropbox.com/sh/4xds64j6atmi68j/AADEZ94lZruhG5_JxpWee_g3a/openvqa/ckpts/ckpt_dataset_pt1_m143 -O ckpt_dataset_pt1_m143.zip
wget https://www.dropbox.com/sh/4xds64j6atmi68j/AACvX8z0Zo4yYFjL7n78D0Lha/openvqa/ckpts/ckpt_dataset_pt1_m144 -O ckpt_dataset_pt1_m144.zip
wget https://www.dropbox.com/sh/4xds64j6atmi68j/AACHTiCycTylLQU-k7WCYl7qa/openvqa/ckpts/ckpt_dataset_pt1_m145 -O ckpt_dataset_pt1_m145.zip
wget https://www.dropbox.com/sh/4xds64j6atmi68j/AABCtrXFgqX3-_NB4OCZkRysa/openvqa/ckpts/ckpt_dataset_pt1_m146 -O ckpt_dataset_pt1_m146.zip
wget https://www.dropbox.com/sh/4xds64j6atmi68j/AACJKAWTJbixRHz-ezyiGRmla/openvqa/ckpts/ckpt_dataset_pt1_m147 -O ckpt_dataset_pt1_m147.zip
wget https://www.dropbox.com/sh/4xds64j6atmi68j/AACFpXptytpO3ZO_yi-BhkxOa/openvqa/ckpts/ckpt_dataset_pt1_m148 -O ckpt_dataset_pt1_m148.zip
wget https://www.dropbox.com/sh/4xds64j6atmi68j/AAAT2ETgKa7NdzpvP0Y6JKu9a/openvqa/ckpts/ckpt_dataset_pt1_m149 -O ckpt_dataset_pt1_m149.zip
wget https://www.dropbox.com/sh/4xds64j6atmi68j/AADbkruG9ZN9QxHkpzAk-Efpa/openvqa/ckpts/ckpt_dataset_pt1_m151 -O ckpt_dataset_pt1_m151.zip
wget https://www.dropbox.com/sh/4xds64j6atmi68j/AABhdo32drV3k85FBkdxqmsba/openvqa/ckpts/ckpt_dataset_pt1_m152 -O ckpt_dataset_pt1_m152.zip
wget https://www.dropbox.com/sh/4xds64j6atmi68j/AACUtMNzJLov5FJBgW-hipTka/openvqa/ckpts/ckpt_dataset_pt1_m153 -O ckpt_dataset_pt1_m153.zip
wget https://www.dropbox.com/sh/4xds64j6atmi68j/AABpGITgVvHCanCF4Vd0iqCIa/openvqa/ckpts/ckpt_dataset_pt1_m154 -O ckpt_dataset_pt1_m154.zip
wget https://www.dropbox.com/sh/4xds64j6atmi68j/AAAm6CMAbiWMFH_d1SyoDmrla/openvqa/ckpts/ckpt_dataset_pt1_m155 -O ckpt_dataset_pt1_m155.zip
wget https://www.dropbox.com/sh/4xds64j6atmi68j/AADzXVyPBKI6qO241Nzj8V-Ia/openvqa/ckpts/ckpt_dataset_pt1_m156 -O ckpt_dataset_pt1_m156.zip
wget https://www.dropbox.com/sh/4xds64j6atmi68j/AABaF3k7EI42cOZm3ZyB_nE6a/openvqa/ckpts/ckpt_dataset_pt1_m157 -O ckpt_dataset_pt1_m157.zip
wget https://www.dropbox.com/sh/4xds64j6atmi68j/AABb0ZuzQEZFs2pwGBzD5umua/openvqa/ckpts/ckpt_dataset_pt1_m158 -O ckpt_dataset_pt1_m158.zip
wget https://www.dropbox.com/sh/4xds64j6atmi68j/AAANlpIEXx5F7Zkq-WcgaF_2a/openvqa/ckpts/ckpt_dataset_pt1_m159 -O ckpt_dataset_pt1_m159.zip
wget https://www.dropbox.com/sh/4xds64j6atmi68j/AADzZ93sCQYCX8STgHa808NUa/openvqa/ckpts/ckpt_dataset_pt1_m161 -O ckpt_dataset_pt1_m161.zip
wget https://www.dropbox.com/sh/4xds64j6atmi68j/AADBLOfbnrkbRDrwuY7MClr5a/openvqa/ckpts/ckpt_dataset_pt1_m162 -O ckpt_dataset_pt1_m162.zip
wget https://www.dropbox.com/sh/4xds64j6atmi68j/AADAz89GJMF3wD9nbchs4FZQa/openvqa/ckpts/ckpt_dataset_pt1_m163 -O ckpt_dataset_pt1_m163.zip
wget https://www.dropbox.com/sh/4xds64j6atmi68j/AADWo9q5v96r7IwBuATCcqDca/openvqa/ckpts/ckpt_dataset_pt1_m164 -O ckpt_dataset_pt1_m164.zip
wget https://www.dropbox.com/sh/4xds64j6atmi68j/AACVTVsbjJhcdy-SnT9ViGS0a/openvqa/ckpts/ckpt_dataset_pt1_m165 -O ckpt_dataset_pt1_m165.zip
wget https://www.dropbox.com/sh/4xds64j6atmi68j/AADtxzKX7MUlPmdrfT-9ZLTPa/openvqa/ckpts/ckpt_dataset_pt1_m166 -O ckpt_dataset_pt1_m166.zip
wget https://www.dropbox.com/sh/4xds64j6atmi68j/AACumWGmf6sykruYvnNR6-Oja/openvqa/ckpts/ckpt_dataset_pt1_m167 -O ckpt_dataset_pt1_m167.zip
wget https://www.dropbox.com/sh/4xds64j6atmi68j/AADIrmvg0SF3tatE9rKJ08v6a/openvqa/ckpts/ckpt_dataset_pt1_m168 -O ckpt_dataset_pt1_m168.zip
wget https://www.dropbox.com/sh/4xds64j6atmi68j/AAB-H2VN0QmMF9guWNOKpeKMa/openvqa/ckpts/ckpt_dataset_pt1_m169 -O ckpt_dataset_pt1_m169.zip
wget https://www.dropbox.com/sh/4xds64j6atmi68j/AABkXQJmyu1DJJ8a2Mpl8DPOa/openvqa/ckpts/ckpt_dataset_pt1_m171 -O ckpt_dataset_pt1_m171.zip
wget https://www.dropbox.com/sh/4xds64j6atmi68j/AACM3Qkn8MWvOTFQ6edjWkk-a/openvqa/ckpts/ckpt_dataset_pt1_m172 -O ckpt_dataset_pt1_m172.zip
wget https://www.dropbox.com/sh/4xds64j6atmi68j/AAAtCOtoujzdQEIN1HQ4b0yFa/openvqa/ckpts/ckpt_dataset_pt1_m173 -O ckpt_dataset_pt1_m173.zip
wget https://www.dropbox.com/sh/4xds64j6atmi68j/AADtn3sQpm1KpRUcWoBccfC9a/openvqa/ckpts/ckpt_dataset_pt1_m174 -O ckpt_dataset_pt1_m174.zip
wget https://www.dropbox.com/sh/4xds64j6atmi68j/AAC4Ig4gM71FN_B94cDdr3lba/openvqa/ckpts/ckpt_dataset_pt1_m175 -O ckpt_dataset_pt1_m175.zip
wget https://www.dropbox.com/sh/4xds64j6atmi68j/AAC3jtCTmEzNuQyDXYsL6rnZa/openvqa/ckpts/ckpt_dataset_pt1_m176 -O ckpt_dataset_pt1_m176.zip
wget https://www.dropbox.com/sh/4xds64j6atmi68j/AACiGKeV69Z2Y4ODSIiJCeFna/openvqa/ckpts/ckpt_dataset_pt1_m177 -O ckpt_dataset_pt1_m177.zip
wget https://www.dropbox.com/sh/4xds64j6atmi68j/AACYLI0ZEV6HMcAvpeMvtMe6a/openvqa/ckpts/ckpt_dataset_pt1_m178 -O ckpt_dataset_pt1_m178.zip
wget https://www.dropbox.com/sh/4xds64j6atmi68j/AABlH7KSgjEA5BKjwkzzq8Xva/openvqa/ckpts/ckpt_dataset_pt1_m179 -O ckpt_dataset_pt1_m179.zip
wget https://www.dropbox.com/sh/4xds64j6atmi68j/AABXgKhQF9XocG7ze2r-WasHa/openvqa/ckpts/ckpt_dataset_pt1_m181 -O ckpt_dataset_pt1_m181.zip
wget https://www.dropbox.com/sh/4xds64j6atmi68j/AADENBwDLS9XpJBaPLtfs7Bma/openvqa/ckpts/ckpt_dataset_pt1_m182 -O ckpt_dataset_pt1_m182.zip
wget https://www.dropbox.com/sh/4xds64j6atmi68j/AAB1R-jtHAm3OI3s95kmA_Gya/openvqa/ckpts/ckpt_dataset_pt1_m183 -O ckpt_dataset_pt1_m183.zip
wget https://www.dropbox.com/sh/4xds64j6atmi68j/AAAy1IwXLl61qe6GBcNHM0zna/openvqa/ckpts/ckpt_dataset_pt1_m184 -O ckpt_dataset_pt1_m184.zip
wget https://www.dropbox.com/sh/4xds64j6atmi68j/AADAD5LeJHf4W2T9wUPQmXCva/openvqa/ckpts/ckpt_dataset_pt1_m185 -O ckpt_dataset_pt1_m185.zip
wget https://www.dropbox.com/sh/4xds64j6atmi68j/AAALQEuSQjzjTj9HGiy_oM1oa/openvqa/ckpts/ckpt_dataset_pt1_m186 -O ckpt_dataset_pt1_m186.zip
wget https://www.dropbox.com/sh/4xds64j6atmi68j/AAB9wxu9Gb47U1pseH8tOjFNa/openvqa/ckpts/ckpt_dataset_pt1_m187 -O ckpt_dataset_pt1_m187.zip
wget https://www.dropbox.com/sh/4xds64j6atmi68j/AAD3xyNVt8_diuxvQVAMKRP-a/openvqa/ckpts/ckpt_dataset_pt1_m188 -O ckpt_dataset_pt1_m188.zip
wget https://www.dropbox.com/sh/4xds64j6atmi68j/AAClr6mN2PKTxu4Er13edg2ya/openvqa/ckpts/ckpt_dataset_pt1_m189 -O ckpt_dataset_pt1_m189.zip
wget https://www.dropbox.com/sh/4xds64j6atmi68j/AAB6WM7zK3J91RJ1PLxxZOKSa/openvqa/ckpts/ckpt_dataset_pt1_m191 -O ckpt_dataset_pt1_m191.zip
wget https://www.dropbox.com/sh/4xds64j6atmi68j/AAAnm3vG6IPtg1cPWZjP58nUa/openvqa/ckpts/ckpt_dataset_pt1_m192 -O ckpt_dataset_pt1_m192.zip
wget https://www.dropbox.com/sh/4xds64j6atmi68j/AACqV_XmL1LKwE5-T8Z1iScSa/openvqa/ckpts/ckpt_dataset_pt1_m193 -O ckpt_dataset_pt1_m193.zip
wget https://www.dropbox.com/sh/4xds64j6atmi68j/AAACs_3M1GsUkgolvcAzS0Cma/openvqa/ckpts/ckpt_dataset_pt1_m194 -O ckpt_dataset_pt1_m194.zip
wget https://www.dropbox.com/sh/4xds64j6atmi68j/AADDb6Z5lhF03iPVvEMN7KbMa/openvqa/ckpts/ckpt_dataset_pt1_m195 -O ckpt_dataset_pt1_m195.zip
wget https://www.dropbox.com/sh/4xds64j6atmi68j/AAD2ltsElXXSM9dzGEj4sgsKa/openvqa/ckpts/ckpt_dataset_pt1_m196 -O ckpt_dataset_pt1_m196.zip
wget https://www.dropbox.com/sh/4xds64j6atmi68j/AAA6pct2LVBG5De3JGKZ_rUya/openvqa/ckpts/ckpt_dataset_pt1_m197 -O ckpt_dataset_pt1_m197.zip
wget https://www.dropbox.com/sh/4xds64j6atmi68j/AAAlJhYCvzGkJ1Ltn5VM5ooma/openvqa/ckpts/ckpt_dataset_pt1_m198 -O ckpt_dataset_pt1_m198.zip
wget https://www.dropbox.com/sh/4xds64j6atmi68j/AAD0_OP8y6a4T-O22lwFeRYYa/openvqa/ckpts/ckpt_dataset_pt1_m199 -O ckpt_dataset_pt1_m199.zip
wget https://www.dropbox.com/sh/4xds64j6atmi68j/AADRbGV65mnIguwLCCE5AvkEa/openvqa/ckpts/ckpt_dataset_pt1_m201 -O ckpt_dataset_pt1_m201.zip
wget https://www.dropbox.com/sh/4xds64j6atmi68j/AAA2bYG61dj6OixE8oFWtwVJa/openvqa/ckpts/ckpt_dataset_pt1_m202 -O ckpt_dataset_pt1_m202.zip
wget https://www.dropbox.com/sh/4xds64j6atmi68j/AAD8TRuNWtqOZ2ungj-uRqcEa/openvqa/ckpts/ckpt_dataset_pt1_m203 -O ckpt_dataset_pt1_m203.zip
wget https://www.dropbox.com/sh/4xds64j6atmi68j/AACOCBZlvAd2kaNU7H0aE7Rfa/openvqa/ckpts/ckpt_dataset_pt1_m204 -O ckpt_dataset_pt1_m204.zip
wget https://www.dropbox.com/sh/4xds64j6atmi68j/AAB4J4iCc2QM_lpUdFX4tJcXa/openvqa/ckpts/ckpt_dataset_pt1_m205 -O ckpt_dataset_pt1_m205.zip
wget https://www.dropbox.com/sh/4xds64j6atmi68j/AABa6ZAY_3u_dqALWKQl04wua/openvqa/ckpts/ckpt_dataset_pt1_m206 -O ckpt_dataset_pt1_m206.zip
wget https://www.dropbox.com/sh/4xds64j6atmi68j/AADgZDeZZo-6tTpOP5v8mC7va/openvqa/ckpts/ckpt_dataset_pt1_m207 -O ckpt_dataset_pt1_m207.zip
wget https://www.dropbox.com/sh/4xds64j6atmi68j/AADf0_LOl-xH13WTtK4BUraxa/openvqa/ckpts/ckpt_dataset_pt1_m208 -O ckpt_dataset_pt1_m208.zip
wget https://www.dropbox.com/sh/4xds64j6atmi68j/AADGBS6NFzzBpGgKSOyCSiHZa/openvqa/ckpts/ckpt_dataset_pt1_m209 -O ckpt_dataset_pt1_m209.zip
wget https://www.dropbox.com/sh/4xds64j6atmi68j/AABvTzRl-Hd40a4D_7jQL9Bqa/openvqa/ckpts/ckpt_dataset_pt1_m211 -O ckpt_dataset_pt1_m211.zip
wget https://www.dropbox.com/sh/4xds64j6atmi68j/AADsPb7pd2XSTdN2WDtpYSKoa/openvqa/ckpts/ckpt_dataset_pt1_m212 -O ckpt_dataset_pt1_m212.zip
wget https://www.dropbox.com/sh/4xds64j6atmi68j/AAAs4yn8YdWWFuTQLejeQ0Hka/openvqa/ckpts/ckpt_dataset_pt1_m213 -O ckpt_dataset_pt1_m213.zip
wget https://www.dropbox.com/sh/4xds64j6atmi68j/AABc3PVNZf-elUwRgnhkOf3qa/openvqa/ckpts/ckpt_dataset_pt1_m214 -O ckpt_dataset_pt1_m214.zip
wget https://www.dropbox.com/sh/4xds64j6atmi68j/AADP9a_Yl1Il8wLCF07Fp36ua/openvqa/ckpts/ckpt_dataset_pt1_m215 -O ckpt_dataset_pt1_m215.zip
wget https://www.dropbox.com/sh/4xds64j6atmi68j/AAC19VTeC-12n4Vyb1WqUnjPa/openvqa/ckpts/ckpt_dataset_pt1_m216 -O ckpt_dataset_pt1_m216.zip
wget https://www.dropbox.com/sh/4xds64j6atmi68j/AABy2uKf5MD5xSAm06mIyna9a/openvqa/ckpts/ckpt_dataset_pt1_m217 -O ckpt_dataset_pt1_m217.zip
wget https://www.dropbox.com/sh/4xds64j6atmi68j/AABNAb_2xGgiK1gaWDTIA6Pba/openvqa/ckpts/ckpt_dataset_pt1_m218 -O ckpt_dataset_pt1_m218.zip
wget https://www.dropbox.com/sh/4xds64j6atmi68j/AACuZVjCegKfZmroKJnzOA3Oa/openvqa/ckpts/ckpt_dataset_pt1_m219 -O ckpt_dataset_pt1_m219.zip
wget https://www.dropbox.com/sh/4xds64j6atmi68j/AAALmyII3dH1Whh_WhCKv4_Pa/openvqa/ckpts/ckpt_dataset_pt1_m221 -O ckpt_dataset_pt1_m221.zip
wget https://www.dropbox.com/sh/4xds64j6atmi68j/AABVtKZLZIDlkoxQxEmHL1eca/openvqa/ckpts/ckpt_dataset_pt1_m222 -O ckpt_dataset_pt1_m222.zip
wget https://www.dropbox.com/sh/4xds64j6atmi68j/AACQlTBmILQoz1auG6WosExUa/openvqa/ckpts/ckpt_dataset_pt1_m223 -O ckpt_dataset_pt1_m223.zip
wget https://www.dropbox.com/sh/4xds64j6atmi68j/AACbkdjLXB0NvRl4wcjbkukTa/openvqa/ckpts/ckpt_dataset_pt1_m224 -O ckpt_dataset_pt1_m224.zip
wget https://www.dropbox.com/sh/4xds64j6atmi68j/AACWk76hg7KYpBbVgFiEqsfla/openvqa/ckpts/ckpt_dataset_pt1_m225 -O ckpt_dataset_pt1_m225.zip
wget https://www.dropbox.com/sh/4xds64j6atmi68j/AACnk1aIfAW2xZ0_FAe_h311a/openvqa/ckpts/ckpt_dataset_pt1_m226 -O ckpt_dataset_pt1_m226.zip
wget https://www.dropbox.com/sh/4xds64j6atmi68j/AABtdnsXxp2_cfmDRrEQ18eXa/openvqa/ckpts/ckpt_dataset_pt1_m227 -O ckpt_dataset_pt1_m227.zip
wget https://www.dropbox.com/sh/4xds64j6atmi68j/AADFX3UQsO2Ii-uIgYskGLyoa/openvqa/ckpts/ckpt_dataset_pt1_m228 -O ckpt_dataset_pt1_m228.zip
wget https://www.dropbox.com/sh/4xds64j6atmi68j/AADqSO_Wj0AuQkYGvCkKH6qga/openvqa/ckpts/ckpt_dataset_pt1_m229 -O ckpt_dataset_pt1_m229.zip
wget https://www.dropbox.com/sh/4xds64j6atmi68j/AAB0TBMjyJ2Ku5u11oU73w6na/openvqa/ckpts/ckpt_dataset_pt1_m231 -O ckpt_dataset_pt1_m231.zip
wget https://www.dropbox.com/sh/4xds64j6atmi68j/AAAZsLkSdLX13wCYi1_urjBfa/openvqa/ckpts/ckpt_dataset_pt1_m232 -O ckpt_dataset_pt1_m232.zip
wget https://www.dropbox.com/sh/4xds64j6atmi68j/AABWdCjaxWo7h44d96EtTg7Fa/openvqa/ckpts/ckpt_dataset_pt1_m233 -O ckpt_dataset_pt1_m233.zip
wget https://www.dropbox.com/sh/4xds64j6atmi68j/AACMPj0eYguC_NR8GEY14-1ea/openvqa/ckpts/ckpt_dataset_pt1_m234 -O ckpt_dataset_pt1_m234.zip
wget https://www.dropbox.com/sh/4xds64j6atmi68j/AADIuqCp60bdGdC0PZvCkvwha/openvqa/ckpts/ckpt_dataset_pt1_m235 -O ckpt_dataset_pt1_m235.zip
wget https://www.dropbox.com/sh/4xds64j6atmi68j/AADzYtkiTTfid5i5iL1Z3G0-a/openvqa/ckpts/ckpt_dataset_pt1_m236 -O ckpt_dataset_pt1_m236.zip
wget https://www.dropbox.com/sh/4xds64j6atmi68j/AADaz6M_hIzTluuExOr2jlRXa/openvqa/ckpts/ckpt_dataset_pt1_m237 -O ckpt_dataset_pt1_m237.zip
wget https://www.dropbox.com/sh/4xds64j6atmi68j/AADrtiCZQ-fK_NbbR8c8k5CKa/openvqa/ckpts/ckpt_dataset_pt1_m238 -O ckpt_dataset_pt1_m238.zip
wget https://www.dropbox.com/sh/4xds64j6atmi68j/AADnrui3e-JJC8tIwfdIxhEqa/openvqa/ckpts/ckpt_dataset_pt1_m239 -O ckpt_dataset_pt1_m239.zip
wget https://www.dropbox.com/sh/4xds64j6atmi68j/AABTPs7WmLbQrNpm1jCnVfKaa/openvqa/ckpts/ckpt_dataset_pt2_m1 -O ckpt_dataset_pt2_m1.zip
wget https://www.dropbox.com/sh/4xds64j6atmi68j/AACOGCCbQKstmDneKj3OTyrsa/openvqa/ckpts/ckpt_dataset_pt2_m2 -O ckpt_dataset_pt2_m2.zip
wget https://www.dropbox.com/sh/4xds64j6atmi68j/AAD1mvgW4Gjsq1dB097Vurr4a/openvqa/ckpts/ckpt_dataset_pt2_m3 -O ckpt_dataset_pt2_m3.zip
wget https://www.dropbox.com/sh/4xds64j6atmi68j/AADY5gpGZTI8i6albYLI-PZEa/openvqa/ckpts/ckpt_dataset_pt2_m4 -O ckpt_dataset_pt2_m4.zip
wget https://www.dropbox.com/sh/4xds64j6atmi68j/AADhOSvag2zHzhW36y077K2Xa/openvqa/ckpts/ckpt_dataset_pt2_m5 -O ckpt_dataset_pt2_m5.zip
wget https://www.dropbox.com/sh/4xds64j6atmi68j/AAAVpMUHMritfJhlylE8IK1Ga/openvqa/ckpts/ckpt_dataset_pt2_m6 -O ckpt_dataset_pt2_m6.zip
wget https://www.dropbox.com/sh/4xds64j6atmi68j/AADF7UhydJttUK95IqjnJ_Vaa/openvqa/ckpts/ckpt_dataset_pt2_m7 -O ckpt_dataset_pt2_m7.zip
wget https://www.dropbox.com/sh/4xds64j6atmi68j/AAAWrBmY-rLGAHkc5aGhboHta/openvqa/ckpts/ckpt_dataset_pt2_m8 -O ckpt_dataset_pt2_m8.zip
wget https://www.dropbox.com/sh/4xds64j6atmi68j/AADsdzRG9blQwLcWuvP0OipXa/openvqa/ckpts/ckpt_dataset_pt2_m9 -O ckpt_dataset_pt2_m9.zip
wget https://www.dropbox.com/sh/4xds64j6atmi68j/AABp_kDKwxaPInpVPzhwF8N_a/openvqa/ckpts/ckpt_dataset_pt2_m11 -O ckpt_dataset_pt2_m11.zip
wget https://www.dropbox.com/sh/4xds64j6atmi68j/AAAFVpdR_rCbsXAKwYQ2cSbya/openvqa/ckpts/ckpt_dataset_pt2_m12 -O ckpt_dataset_pt2_m12.zip
wget https://www.dropbox.com/sh/4xds64j6atmi68j/AAAo4Df8CBde_LoOfTX1u1cAa/openvqa/ckpts/ckpt_dataset_pt2_m13 -O ckpt_dataset_pt2_m13.zip
wget https://www.dropbox.com/sh/4xds64j6atmi68j/AABx99mia-bkOLsX4NZgwElwa/openvqa/ckpts/ckpt_dataset_pt2_m14 -O ckpt_dataset_pt2_m14.zip
wget https://www.dropbox.com/sh/4xds64j6atmi68j/AAC-Mw8_AGTYEcM0pu3J9Ms2a/openvqa/ckpts/ckpt_dataset_pt2_m15 -O ckpt_dataset_pt2_m15.zip
wget https://www.dropbox.com/sh/4xds64j6atmi68j/AAAhTXYb49BJoFnrWpOiwA-Za/openvqa/ckpts/ckpt_dataset_pt2_m16 -O ckpt_dataset_pt2_m16.zip
wget https://www.dropbox.com/sh/4xds64j6atmi68j/AACTD5hZthSekIvy9dwgXX1va/openvqa/ckpts/ckpt_dataset_pt2_m17 -O ckpt_dataset_pt2_m17.zip
wget https://www.dropbox.com/sh/4xds64j6atmi68j/AABOndgw-cEayX2eFV0qtppsa/openvqa/ckpts/ckpt_dataset_pt2_m18 -O ckpt_dataset_pt2_m18.zip
wget https://www.dropbox.com/sh/4xds64j6atmi68j/AACZ2IYxFu4Rzq5bv8B3-k9Ja/openvqa/ckpts/ckpt_dataset_pt2_m19 -O ckpt_dataset_pt2_m19.zip
wget https://www.dropbox.com/sh/4xds64j6atmi68j/AADbS0B2dM6A7XN4znpggUdqa/openvqa/ckpts/ckpt_dataset_pt2_m21 -O ckpt_dataset_pt2_m21.zip
wget https://www.dropbox.com/sh/4xds64j6atmi68j/AADy7dzcsjW_Dxpquwwy43fpa/openvqa/ckpts/ckpt_dataset_pt2_m22 -O ckpt_dataset_pt2_m22.zip
wget https://www.dropbox.com/sh/4xds64j6atmi68j/AAAhlcWowktpdOg-gM8dNFD4a/openvqa/ckpts/ckpt_dataset_pt2_m23 -O ckpt_dataset_pt2_m23.zip
wget https://www.dropbox.com/sh/4xds64j6atmi68j/AAB1WquD6456pe3Yr_7hpo7Va/openvqa/ckpts/ckpt_dataset_pt2_m24 -O ckpt_dataset_pt2_m24.zip
wget https://www.dropbox.com/sh/4xds64j6atmi68j/AACBWH9tFG6xAg6FkJ16lz4Ea/openvqa/ckpts/ckpt_dataset_pt2_m25 -O ckpt_dataset_pt2_m25.zip
wget https://www.dropbox.com/sh/4xds64j6atmi68j/AABYeoUGCW2GAnt2ONwOdH9Da/openvqa/ckpts/ckpt_dataset_pt2_m26 -O ckpt_dataset_pt2_m26.zip
wget https://www.dropbox.com/sh/4xds64j6atmi68j/AACjJEMkII2CuYyUaeLTnCcXa/openvqa/ckpts/ckpt_dataset_pt2_m27 -O ckpt_dataset_pt2_m27.zip
wget https://www.dropbox.com/sh/4xds64j6atmi68j/AAC8hYn4tc5oP0Cie9L5h7ila/openvqa/ckpts/ckpt_dataset_pt2_m28 -O ckpt_dataset_pt2_m28.zip
wget https://www.dropbox.com/sh/4xds64j6atmi68j/AACDMGJ0J549sjuZvHPd2Ep7a/openvqa/ckpts/ckpt_dataset_pt2_m29 -O ckpt_dataset_pt2_m29.zip
wget https://www.dropbox.com/sh/4xds64j6atmi68j/AADj_NdMuUl8mC6cQbzAGwtSa/openvqa/ckpts/ckpt_dataset_pt2_m31 -O ckpt_dataset_pt2_m31.zip
wget https://www.dropbox.com/sh/4xds64j6atmi68j/AACzRu0QkZPvzAR-r4nrJyrZa/openvqa/ckpts/ckpt_dataset_pt2_m32 -O ckpt_dataset_pt2_m32.zip
wget https://www.dropbox.com/sh/4xds64j6atmi68j/AACovKC2xUCVmJyIBLUafN22a/openvqa/ckpts/ckpt_dataset_pt2_m33 -O ckpt_dataset_pt2_m33.zip
wget https://www.dropbox.com/sh/4xds64j6atmi68j/AAAklRQpHBoL9kgjyyLo-tXqa/openvqa/ckpts/ckpt_dataset_pt2_m34 -O ckpt_dataset_pt2_m34.zip
wget https://www.dropbox.com/sh/4xds64j6atmi68j/AADWc_DJr57KC_CmMQFIosBja/openvqa/ckpts/ckpt_dataset_pt2_m35 -O ckpt_dataset_pt2_m35.zip
wget https://www.dropbox.com/sh/4xds64j6atmi68j/AABsF-tLag-vO6v7XRMlTvUga/openvqa/ckpts/ckpt_dataset_pt2_m36 -O ckpt_dataset_pt2_m36.zip
wget https://www.dropbox.com/sh/4xds64j6atmi68j/AABmqVl9SQ-RdDmr-0H74z0Ia/openvqa/ckpts/ckpt_dataset_pt2_m37 -O ckpt_dataset_pt2_m37.zip
wget https://www.dropbox.com/sh/4xds64j6atmi68j/AADGoI8Ns1NxgWXgnyGMDxwOa/openvqa/ckpts/ckpt_dataset_pt2_m38 -O ckpt_dataset_pt2_m38.zip
wget https://www.dropbox.com/sh/4xds64j6atmi68j/AABSZRgyDP4YT5ck1Fr0B2GJa/openvqa/ckpts/ckpt_dataset_pt2_m39 -O ckpt_dataset_pt2_m39.zip
wget https://www.dropbox.com/sh/4xds64j6atmi68j/AADQ_jQ5SwwIQ7bjBi9iSjRHa/openvqa/ckpts/ckpt_dataset_pt2_m41 -O ckpt_dataset_pt2_m41.zip
wget https://www.dropbox.com/sh/4xds64j6atmi68j/AACM_on-hH1At0jufMpaL97ca/openvqa/ckpts/ckpt_dataset_pt2_m42 -O ckpt_dataset_pt2_m42.zip
wget https://www.dropbox.com/sh/4xds64j6atmi68j/AADeHdA4S2rN6SXC_k_FOTmRa/openvqa/ckpts/ckpt_dataset_pt2_m43 -O ckpt_dataset_pt2_m43.zip
wget https://www.dropbox.com/sh/4xds64j6atmi68j/AADAVoMm0hgOHFg-Xvu0tjoqa/openvqa/ckpts/ckpt_dataset_pt2_m44 -O ckpt_dataset_pt2_m44.zip
wget https://www.dropbox.com/sh/4xds64j6atmi68j/AABiBxEHzai97gnL1zza3CLda/openvqa/ckpts/ckpt_dataset_pt2_m45 -O ckpt_dataset_pt2_m45.zip
wget https://www.dropbox.com/sh/4xds64j6atmi68j/AACtjFERPRixLYPfQB1M2tiXa/openvqa/ckpts/ckpt_dataset_pt2_m46 -O ckpt_dataset_pt2_m46.zip
wget https://www.dropbox.com/sh/4xds64j6atmi68j/AADC7apdSBG-6dG3iS2_yCW-a/openvqa/ckpts/ckpt_dataset_pt2_m47 -O ckpt_dataset_pt2_m47.zip
wget https://www.dropbox.com/sh/4xds64j6atmi68j/AABZwFLdAS56Yz9wX49kEa7Oa/openvqa/ckpts/ckpt_dataset_pt2_m48 -O ckpt_dataset_pt2_m48.zip
wget https://www.dropbox.com/sh/4xds64j6atmi68j/AAB9Xd9wUD_hASbMSYzNY9noa/openvqa/ckpts/ckpt_dataset_pt2_m49 -O ckpt_dataset_pt2_m49.zip
wget https://www.dropbox.com/sh/4xds64j6atmi68j/AACMKJTiKnGG8K1tG9s5NCoKa/openvqa/ckpts/ckpt_dataset_pt2_m51 -O ckpt_dataset_pt2_m51.zip
wget https://www.dropbox.com/sh/4xds64j6atmi68j/AACYigKieK32Yf5nRVWSuJXja/openvqa/ckpts/ckpt_dataset_pt2_m52 -O ckpt_dataset_pt2_m52.zip
wget https://www.dropbox.com/sh/4xds64j6atmi68j/AABrFvtfiU9_gUYi22W1EtMTa/openvqa/ckpts/ckpt_dataset_pt2_m53 -O ckpt_dataset_pt2_m53.zip
wget https://www.dropbox.com/sh/4xds64j6atmi68j/AABI7Zm1hPpqBgH5Ew_jIFcba/openvqa/ckpts/ckpt_dataset_pt2_m54 -O ckpt_dataset_pt2_m54.zip
wget https://www.dropbox.com/sh/4xds64j6atmi68j/AAB6lfWiMjUNY9pE8kVxEHNAa/openvqa/ckpts/ckpt_dataset_pt2_m55 -O ckpt_dataset_pt2_m55.zip
wget https://www.dropbox.com/sh/4xds64j6atmi68j/AABICnApQa1_slzJKQayKe6va/openvqa/ckpts/ckpt_dataset_pt2_m56 -O ckpt_dataset_pt2_m56.zip
wget https://www.dropbox.com/sh/4xds64j6atmi68j/AADgT6JjgyN6uN2cVUKp1cdya/openvqa/ckpts/ckpt_dataset_pt2_m57 -O ckpt_dataset_pt2_m57.zip
wget https://www.dropbox.com/sh/4xds64j6atmi68j/AAAa2_jOvhtemAbk4T7hYQx2a/openvqa/ckpts/ckpt_dataset_pt2_m58 -O ckpt_dataset_pt2_m58.zip
wget https://www.dropbox.com/sh/4xds64j6atmi68j/AAAo4j2nD6I3KUQEtlORf9aka/openvqa/ckpts/ckpt_dataset_pt2_m59 -O ckpt_dataset_pt2_m59.zip
wget https://www.dropbox.com/sh/4xds64j6atmi68j/AACH5O4Yj5Y-4nTQ3ixZTv5na/openvqa/ckpts/ckpt_dataset_pt2_m61 -O ckpt_dataset_pt2_m61.zip
wget https://www.dropbox.com/sh/4xds64j6atmi68j/AADt_MzTDOoZO7hJKFLgvsxJa/openvqa/ckpts/ckpt_dataset_pt2_m62 -O ckpt_dataset_pt2_m62.zip
wget https://www.dropbox.com/sh/4xds64j6atmi68j/AADW9doBxaLspkU7ENBJxQEGa/openvqa/ckpts/ckpt_dataset_pt2_m63 -O ckpt_dataset_pt2_m63.zip
wget https://www.dropbox.com/sh/4xds64j6atmi68j/AACMMxNlEBzSr65HJiyLL28Ba/openvqa/ckpts/ckpt_dataset_pt2_m64 -O ckpt_dataset_pt2_m64.zip
wget https://www.dropbox.com/sh/4xds64j6atmi68j/AACjhYo0HasVZxS7W2h0EcQra/openvqa/ckpts/ckpt_dataset_pt2_m65 -O ckpt_dataset_pt2_m65.zip
wget https://www.dropbox.com/sh/4xds64j6atmi68j/AACR2ZfX49XPszDa9MNcKc1sa/openvqa/ckpts/ckpt_dataset_pt2_m66 -O ckpt_dataset_pt2_m66.zip
wget https://www.dropbox.com/sh/4xds64j6atmi68j/AADS9L6V_dqi-sL7GVN3kloNa/openvqa/ckpts/ckpt_dataset_pt2_m67 -O ckpt_dataset_pt2_m67.zip
wget https://www.dropbox.com/sh/4xds64j6atmi68j/AACuqxPD5BeTmU7-Gt30e1Dla/openvqa/ckpts/ckpt_dataset_pt2_m68 -O ckpt_dataset_pt2_m68.zip
wget https://www.dropbox.com/sh/4xds64j6atmi68j/AAC2PRVX5XUW1KeE2cd42lq5a/openvqa/ckpts/ckpt_dataset_pt2_m69 -O ckpt_dataset_pt2_m69.zip
wget https://www.dropbox.com/sh/4xds64j6atmi68j/AAAAMSdryYo7BbuhnAysFg28a/openvqa/ckpts/ckpt_dataset_pt2_m71 -O ckpt_dataset_pt2_m71.zip
wget https://www.dropbox.com/sh/4xds64j6atmi68j/AAD9piMAxdKv5r6eNJbI1Vwpa/openvqa/ckpts/ckpt_dataset_pt2_m72 -O ckpt_dataset_pt2_m72.zip
wget https://www.dropbox.com/sh/4xds64j6atmi68j/AADs5AQgPdcJdoCDy0rSyYFea/openvqa/ckpts/ckpt_dataset_pt2_m73 -O ckpt_dataset_pt2_m73.zip
wget https://www.dropbox.com/sh/4xds64j6atmi68j/AADye_HI3ph6ty815hu_iAdEa/openvqa/ckpts/ckpt_dataset_pt2_m74 -O ckpt_dataset_pt2_m74.zip
wget https://www.dropbox.com/sh/4xds64j6atmi68j/AACUaIJ-9V0Ul4-UQi4eRn_Pa/openvqa/ckpts/ckpt_dataset_pt2_m75 -O ckpt_dataset_pt2_m75.zip
wget https://www.dropbox.com/sh/4xds64j6atmi68j/AADjgmHORiB2iwiyMGeLkCN5a/openvqa/ckpts/ckpt_dataset_pt2_m76 -O ckpt_dataset_pt2_m76.zip
wget https://www.dropbox.com/sh/4xds64j6atmi68j/AAA8u6iYxqy85MPv0hs8gp2ta/openvqa/ckpts/ckpt_dataset_pt2_m77 -O ckpt_dataset_pt2_m77.zip
wget https://www.dropbox.com/sh/4xds64j6atmi68j/AABZ1j1XTxtlLXMLhKQZFh68a/openvqa/ckpts/ckpt_dataset_pt2_m78 -O ckpt_dataset_pt2_m78.zip
wget https://www.dropbox.com/sh/4xds64j6atmi68j/AADDPM3CnBy40CLAfpxg0uUsa/openvqa/ckpts/ckpt_dataset_pt2_m79 -O ckpt_dataset_pt2_m79.zip
wget https://www.dropbox.com/sh/4xds64j6atmi68j/AADxuwH2QeZIOGgHuiKRc03Qa/openvqa/ckpts/ckpt_dataset_pt2_m81 -O ckpt_dataset_pt2_m81.zip
wget https://www.dropbox.com/sh/4xds64j6atmi68j/AAAtJHNwr9AbKUWxUzc96YnXa/openvqa/ckpts/ckpt_dataset_pt2_m82 -O ckpt_dataset_pt2_m82.zip
wget https://www.dropbox.com/sh/4xds64j6atmi68j/AAC3aasPXel-Bw6Ev7rNyM2ta/openvqa/ckpts/ckpt_dataset_pt2_m83 -O ckpt_dataset_pt2_m83.zip
wget https://www.dropbox.com/sh/4xds64j6atmi68j/AAAnzUXAdGSqBGGocPzkUW7fa/openvqa/ckpts/ckpt_dataset_pt2_m84 -O ckpt_dataset_pt2_m84.zip
wget https://www.dropbox.com/sh/4xds64j6atmi68j/AAAnCkgvmoB4lypmy-CmfJ_Ca/openvqa/ckpts/ckpt_dataset_pt2_m85 -O ckpt_dataset_pt2_m85.zip
wget https://www.dropbox.com/sh/4xds64j6atmi68j/AAC9aaMcxb4G6MS1izchvRsPa/openvqa/ckpts/ckpt_dataset_pt2_m86 -O ckpt_dataset_pt2_m86.zip
wget https://www.dropbox.com/sh/4xds64j6atmi68j/AADAs0eOYVQJdikVy5Im8Xala/openvqa/ckpts/ckpt_dataset_pt2_m87 -O ckpt_dataset_pt2_m87.zip
wget https://www.dropbox.com/sh/4xds64j6atmi68j/AAC7dkyunGt8AlQVcBE3B4e_a/openvqa/ckpts/ckpt_dataset_pt2_m88 -O ckpt_dataset_pt2_m88.zip
wget https://www.dropbox.com/sh/4xds64j6atmi68j/AAD7U3qKK2YT1oDZE50Y7ppca/openvqa/ckpts/ckpt_dataset_pt2_m89 -O ckpt_dataset_pt2_m89.zip
wget https://www.dropbox.com/sh/4xds64j6atmi68j/AAA_aRjuij7V4vTydwTpZTAQa/openvqa/ckpts/ckpt_dataset_pt2_m91 -O ckpt_dataset_pt2_m91.zip
wget https://www.dropbox.com/sh/4xds64j6atmi68j/AABlaF9pN8Mt9C4dmRbR_8TBa/openvqa/ckpts/ckpt_dataset_pt2_m92 -O ckpt_dataset_pt2_m92.zip
wget https://www.dropbox.com/sh/4xds64j6atmi68j/AADrKK4JUEPiTPhTpZR3bi7za/openvqa/ckpts/ckpt_dataset_pt2_m93 -O ckpt_dataset_pt2_m93.zip
wget https://www.dropbox.com/sh/4xds64j6atmi68j/AADfnIyYW2cuaFMa_ySjkIf8a/openvqa/ckpts/ckpt_dataset_pt2_m94 -O ckpt_dataset_pt2_m94.zip
wget https://www.dropbox.com/sh/4xds64j6atmi68j/AAD16OrV2as6yA2acQNHmJ2-a/openvqa/ckpts/ckpt_dataset_pt2_m95 -O ckpt_dataset_pt2_m95.zip
wget https://www.dropbox.com/sh/4xds64j6atmi68j/AABMG7NM5abZHn8I9zJjU118a/openvqa/ckpts/ckpt_dataset_pt2_m96 -O ckpt_dataset_pt2_m96.zip
wget https://www.dropbox.com/sh/4xds64j6atmi68j/AAAoPtUIERDZiPDHSQYOqYDpa/openvqa/ckpts/ckpt_dataset_pt2_m97 -O ckpt_dataset_pt2_m97.zip
wget https://www.dropbox.com/sh/4xds64j6atmi68j/AADYQMWqxuPewDZoATH-RXFfa/openvqa/ckpts/ckpt_dataset_pt2_m98 -O ckpt_dataset_pt2_m98.zip
wget https://www.dropbox.com/sh/4xds64j6atmi68j/AAAs4gQDG4-nWlQ6J4sPwOoBa/openvqa/ckpts/ckpt_dataset_pt2_m99 -O ckpt_dataset_pt2_m99.zip
wget https://www.dropbox.com/sh/4xds64j6atmi68j/AACYsYvrh0qyAugna1F8tWhaa/openvqa/ckpts/ckpt_dataset_pt2_m101 -O ckpt_dataset_pt2_m101.zip
wget https://www.dropbox.com/sh/4xds64j6atmi68j/AABrlFIUYKcRX_VAKgtZ_jBfa/openvqa/ckpts/ckpt_dataset_pt2_m102 -O ckpt_dataset_pt2_m102.zip
wget https://www.dropbox.com/sh/4xds64j6atmi68j/AADWGPN_79w1VHNYtAP2hAzTa/openvqa/ckpts/ckpt_dataset_pt2_m103 -O ckpt_dataset_pt2_m103.zip
wget https://www.dropbox.com/sh/4xds64j6atmi68j/AAClKpuEoquqPaC3Q91SCDbGa/openvqa/ckpts/ckpt_dataset_pt2_m104 -O ckpt_dataset_pt2_m104.zip
wget https://www.dropbox.com/sh/4xds64j6atmi68j/AABt3J_ITQqLMac8FbzpUEPQa/openvqa/ckpts/ckpt_dataset_pt2_m105 -O ckpt_dataset_pt2_m105.zip
wget https://www.dropbox.com/sh/4xds64j6atmi68j/AADs2DEe2IXBKQbgH7XkpVwRa/openvqa/ckpts/ckpt_dataset_pt2_m106 -O ckpt_dataset_pt2_m106.zip
wget https://www.dropbox.com/sh/4xds64j6atmi68j/AAAifKAa1RT6UnEQEEiaDqWPa/openvqa/ckpts/ckpt_dataset_pt2_m107 -O ckpt_dataset_pt2_m107.zip
wget https://www.dropbox.com/sh/4xds64j6atmi68j/AABPio7i8tnQuAq-30Lkms0_a/openvqa/ckpts/ckpt_dataset_pt2_m108 -O ckpt_dataset_pt2_m108.zip
wget https://www.dropbox.com/sh/4xds64j6atmi68j/AABE0Ybx3XwAXw6cRQhRfkgOa/openvqa/ckpts/ckpt_dataset_pt2_m109 -O ckpt_dataset_pt2_m109.zip
wget https://www.dropbox.com/sh/4xds64j6atmi68j/AACQdKWkWrlQ1hhfNTlQUR-Ga/openvqa/ckpts/ckpt_dataset_pt2_m111 -O ckpt_dataset_pt2_m111.zip
wget https://www.dropbox.com/sh/4xds64j6atmi68j/AADsTbVjehfYHPn73q__Hs3ta/openvqa/ckpts/ckpt_dataset_pt2_m112 -O ckpt_dataset_pt2_m112.zip
wget https://www.dropbox.com/sh/4xds64j6atmi68j/AACzWzqkZZiteDAKdtMEzxILa/openvqa/ckpts/ckpt_dataset_pt2_m113 -O ckpt_dataset_pt2_m113.zip
wget https://www.dropbox.com/sh/4xds64j6atmi68j/AAAe2GOkC-2Z_-7PdTNZpoPca/openvqa/ckpts/ckpt_dataset_pt2_m114 -O ckpt_dataset_pt2_m114.zip
wget https://www.dropbox.com/sh/4xds64j6atmi68j/AACiSLKdCBelxsoHgPuzwjDba/openvqa/ckpts/ckpt_dataset_pt2_m115 -O ckpt_dataset_pt2_m115.zip
wget https://www.dropbox.com/sh/4xds64j6atmi68j/AABhfME9BJSd39Gkq-WUvVICa/openvqa/ckpts/ckpt_dataset_pt2_m116 -O ckpt_dataset_pt2_m116.zip
wget https://www.dropbox.com/sh/4xds64j6atmi68j/AADDe72vwPsS2LNdqe4msbzXa/openvqa/ckpts/ckpt_dataset_pt2_m117 -O ckpt_dataset_pt2_m117.zip
wget https://www.dropbox.com/sh/4xds64j6atmi68j/AAC_sjljtAYZIyuK_dVOYItma/openvqa/ckpts/ckpt_dataset_pt2_m118 -O ckpt_dataset_pt2_m118.zip
wget https://www.dropbox.com/sh/4xds64j6atmi68j/AAATRseOK21G0cx2OH3YeBG8a/openvqa/ckpts/ckpt_dataset_pt2_m119 -O ckpt_dataset_pt2_m119.zip
wget https://www.dropbox.com/sh/4xds64j6atmi68j/AABFH9b4aKP6KbsEf8n6ZBfBa/openvqa/ckpts/ckpt_dataset_pt3_m1 -O ckpt_dataset_pt3_m1.zip
wget https://www.dropbox.com/sh/4xds64j6atmi68j/AAA6LylZkQf4St9fxIqGqosOa/openvqa/ckpts/ckpt_dataset_pt3_m2 -O ckpt_dataset_pt3_m2.zip
wget https://www.dropbox.com/sh/4xds64j6atmi68j/AABiVI4Y-c6jcc7T4KwGY5pla/openvqa/ckpts/ckpt_dataset_pt3_m3 -O ckpt_dataset_pt3_m3.zip
wget https://www.dropbox.com/sh/4xds64j6atmi68j/AADdNgNN9VG1ke-rYneYl8wJa/openvqa/ckpts/ckpt_dataset_pt3_m4 -O ckpt_dataset_pt3_m4.zip
wget https://www.dropbox.com/sh/4xds64j6atmi68j/AABbdBexavnUfc_KZA-P1pena/openvqa/ckpts/ckpt_dataset_pt3_m5 -O ckpt_dataset_pt3_m5.zip
wget https://www.dropbox.com/sh/4xds64j6atmi68j/AAAZke_EwLJGuQVszGoSY06na/openvqa/ckpts/ckpt_dataset_pt3_m6 -O ckpt_dataset_pt3_m6.zip
wget https://www.dropbox.com/sh/4xds64j6atmi68j/AAA1vc9NFwFZnMeuAVj2Vdgba/openvqa/ckpts/ckpt_dataset_pt3_m7 -O ckpt_dataset_pt3_m7.zip
wget https://www.dropbox.com/sh/4xds64j6atmi68j/AADeZpYIX3U3Nl-gZOW-aopZa/openvqa/ckpts/ckpt_dataset_pt3_m11 -O ckpt_dataset_pt3_m11.zip
wget https://www.dropbox.com/sh/4xds64j6atmi68j/AADPi-2-sOtGk5eLznDs0acNa/openvqa/ckpts/ckpt_dataset_pt3_m12 -O ckpt_dataset_pt3_m12.zip
wget https://www.dropbox.com/sh/4xds64j6atmi68j/AAC-feLFAXebrjSCWUsk65INa/openvqa/ckpts/ckpt_dataset_pt3_m13 -O ckpt_dataset_pt3_m13.zip
wget https://www.dropbox.com/sh/4xds64j6atmi68j/AABkq8CSlb6tFbU-FyE2rtToa/openvqa/ckpts/ckpt_dataset_pt3_m14 -O ckpt_dataset_pt3_m14.zip
wget https://www.dropbox.com/sh/4xds64j6atmi68j/AADL1pU9WiVSi3NLU9gU-bb4a/openvqa/ckpts/ckpt_dataset_pt3_m15 -O ckpt_dataset_pt3_m15.zip
wget https://www.dropbox.com/sh/4xds64j6atmi68j/AADmVb1MbzNSUKjLXjUv71j8a/openvqa/ckpts/ckpt_dataset_pt3_m16 -O ckpt_dataset_pt3_m16.zip
wget https://www.dropbox.com/sh/4xds64j6atmi68j/AAAKnqnQ4euSRjXGL3SfXFkSa/openvqa/ckpts/ckpt_dataset_pt3_m17 -O ckpt_dataset_pt3_m17.zip
wget https://www.dropbox.com/sh/4xds64j6atmi68j/AAC_DBleQiAcIpOa8sWNVX11a/openvqa/ckpts/ckpt_dataset_pt3_m18 -O ckpt_dataset_pt3_m18.zip
wget https://www.dropbox.com/sh/4xds64j6atmi68j/AACyvlrDXhM6vrTzA9-xf150a/openvqa/ckpts/ckpt_dataset_pt3_m19 -O ckpt_dataset_pt3_m19.zip
wget https://www.dropbox.com/sh/4xds64j6atmi68j/AADDIeD1CaonbaBDcRQviSTJa/openvqa/ckpts/ckpt_dataset_pt3_m21 -O ckpt_dataset_pt3_m21.zip
wget https://www.dropbox.com/sh/4xds64j6atmi68j/AADvSBpMbytD_L2qHMcUW4Kwa/openvqa/ckpts/ckpt_dataset_pt3_m22 -O ckpt_dataset_pt3_m22.zip
wget https://www.dropbox.com/sh/4xds64j6atmi68j/AACq3I7H_-yAfWlb36URnLVla/openvqa/ckpts/ckpt_dataset_pt3_m23 -O ckpt_dataset_pt3_m23.zip
wget https://www.dropbox.com/sh/4xds64j6atmi68j/AAD8H2c6Q-ySMMvfxkMLCy-Wa/openvqa/ckpts/ckpt_dataset_pt3_m24 -O ckpt_dataset_pt3_m24.zip
wget https://www.dropbox.com/sh/4xds64j6atmi68j/AACSnABYshpvwe_94MhEHiE-a/openvqa/ckpts/ckpt_dataset_pt3_m25 -O ckpt_dataset_pt3_m25.zip
wget https://www.dropbox.com/sh/4xds64j6atmi68j/AACcPoLX_wisY85g0JYrhTOJa/openvqa/ckpts/ckpt_dataset_pt3_m26 -O ckpt_dataset_pt3_m26.zip
wget https://www.dropbox.com/sh/4xds64j6atmi68j/AABuYAyOmPTHh1seI5JV4hhRa/openvqa/ckpts/ckpt_dataset_pt3_m27 -O ckpt_dataset_pt3_m27.zip
wget https://www.dropbox.com/sh/4xds64j6atmi68j/AACSmCj3EDLQYVkDm8g8IQKsa/openvqa/ckpts/ckpt_dataset_pt3_m28 -O ckpt_dataset_pt3_m28.zip
wget https://www.dropbox.com/sh/4xds64j6atmi68j/AACPSCzmS4SYgutYCzqfoBS6a/openvqa/ckpts/ckpt_dataset_pt3_m29 -O ckpt_dataset_pt3_m29.zip
wget https://www.dropbox.com/sh/4xds64j6atmi68j/AACP5N9-dD_WNcZe34g_tdqVa/openvqa/ckpts/ckpt_dataset_pt3_m31 -O ckpt_dataset_pt3_m31.zip
wget https://www.dropbox.com/sh/4xds64j6atmi68j/AADdr7QtMJdeYpCGdZSDqo2Ma/openvqa/ckpts/ckpt_dataset_pt3_m32 -O ckpt_dataset_pt3_m32.zip
wget https://www.dropbox.com/sh/4xds64j6atmi68j/AACK5ivGlKZ2-mARRj5QvdEQa/openvqa/ckpts/ckpt_dataset_pt3_m33 -O ckpt_dataset_pt3_m33.zip
wget https://www.dropbox.com/sh/4xds64j6atmi68j/AABa3mxvtBKc53WzTpxk7kPUa/openvqa/ckpts/ckpt_dataset_pt3_m34 -O ckpt_dataset_pt3_m34.zip
wget https://www.dropbox.com/sh/4xds64j6atmi68j/AAABrBavwuPG-T-mtGTiHTv-a/openvqa/ckpts/ckpt_dataset_pt3_m35 -O ckpt_dataset_pt3_m35.zip
wget https://www.dropbox.com/sh/4xds64j6atmi68j/AAB0EC2auI_ML2eS0k0Tz0uJa/openvqa/ckpts/ckpt_dataset_pt3_m36 -O ckpt_dataset_pt3_m36.zip
wget https://www.dropbox.com/sh/4xds64j6atmi68j/AABBvoxbWdNQDDjbLLlh3uJWa/openvqa/ckpts/ckpt_dataset_pt3_m37 -O ckpt_dataset_pt3_m37.zip
wget https://www.dropbox.com/sh/4xds64j6atmi68j/AAAOHgrqt6AKFc4TN2q5YnTHa/openvqa/ckpts/ckpt_dataset_pt3_m38 -O ckpt_dataset_pt3_m38.zip
wget https://www.dropbox.com/sh/4xds64j6atmi68j/AAAplX7yv213PTyu593Mtl2Ma/openvqa/ckpts/ckpt_dataset_pt3_m39 -O ckpt_dataset_pt3_m39.zip
wget https://www.dropbox.com/sh/4xds64j6atmi68j/AAD2_JUU1qzjxKHBrZDI1OoIa/openvqa/ckpts/ckpt_dataset_pt3_m41 -O ckpt_dataset_pt3_m41.zip
wget https://www.dropbox.com/sh/4xds64j6atmi68j/AAD8KHJO9EQqyeB35RxNlffja/openvqa/ckpts/ckpt_dataset_pt3_m42 -O ckpt_dataset_pt3_m42.zip
wget https://www.dropbox.com/sh/4xds64j6atmi68j/AACO9MW5V-oUNisOaXKqAYVPa/openvqa/ckpts/ckpt_dataset_pt3_m43 -O ckpt_dataset_pt3_m43.zip
wget https://www.dropbox.com/sh/4xds64j6atmi68j/AABC-3MouaHz1tLlOZ86TBNHa/openvqa/ckpts/ckpt_dataset_pt3_m44 -O ckpt_dataset_pt3_m44.zip
wget https://www.dropbox.com/sh/4xds64j6atmi68j/AABMJxgEAYZSsY72TdbSzAL0a/openvqa/ckpts/ckpt_dataset_pt3_m45 -O ckpt_dataset_pt3_m45.zip
wget https://www.dropbox.com/sh/4xds64j6atmi68j/AACYHANKfoZmszesx3lU_0Mea/openvqa/ckpts/ckpt_dataset_pt3_m46 -O ckpt_dataset_pt3_m46.zip
wget https://www.dropbox.com/sh/4xds64j6atmi68j/AADnv48JOHYTVFsGfZGv1lPna/openvqa/ckpts/ckpt_dataset_pt3_m47 -O ckpt_dataset_pt3_m47.zip
wget https://www.dropbox.com/sh/4xds64j6atmi68j/AAD8f2vurEZVY7fyCX4cnG7qa/openvqa/ckpts/ckpt_dataset_pt3_m48 -O ckpt_dataset_pt3_m48.zip
wget https://www.dropbox.com/sh/4xds64j6atmi68j/AACyBScGhFHJMgZvJfhoLo5ba/openvqa/ckpts/ckpt_dataset_pt3_m49 -O ckpt_dataset_pt3_m49.zip
wget https://www.dropbox.com/sh/4xds64j6atmi68j/AAC6GHXBqs005OqcPZkaAbkLa/openvqa/ckpts/ckpt_dataset_pt3_m51 -O ckpt_dataset_pt3_m51.zip
wget https://www.dropbox.com/sh/4xds64j6atmi68j/AAChK3lY6EPfG0LoWZkKf5D0a/openvqa/ckpts/ckpt_dataset_pt3_m52 -O ckpt_dataset_pt3_m52.zip
wget https://www.dropbox.com/sh/4xds64j6atmi68j/AAC-2WCXpP7xVICLdL68aho0a/openvqa/ckpts/ckpt_dataset_pt3_m53 -O ckpt_dataset_pt3_m53.zip
wget https://www.dropbox.com/sh/4xds64j6atmi68j/AAD1ETH0B0AXOTkQy1WdzfM_a/openvqa/ckpts/ckpt_dataset_pt3_m54 -O ckpt_dataset_pt3_m54.zip
wget https://www.dropbox.com/sh/4xds64j6atmi68j/AACwZkVMKTDUYpkPx1r7K2Efa/openvqa/ckpts/ckpt_dataset_pt3_m55 -O ckpt_dataset_pt3_m55.zip
wget https://www.dropbox.com/sh/4xds64j6atmi68j/AAAL_dkdCCantIDwkuLaaZCEa/openvqa/ckpts/ckpt_dataset_pt3_m56 -O ckpt_dataset_pt3_m56.zip
wget https://www.dropbox.com/sh/4xds64j6atmi68j/AACuoXPRgJAewgeRouWH28Vma/openvqa/ckpts/ckpt_dataset_pt3_m57 -O ckpt_dataset_pt3_m57.zip
wget https://www.dropbox.com/sh/4xds64j6atmi68j/AAA40c0u_N6X5X1p2Hav5VUYa/openvqa/ckpts/ckpt_dataset_pt3_m58 -O ckpt_dataset_pt3_m58.zip
wget https://www.dropbox.com/sh/4xds64j6atmi68j/AABatNxMIuFdkYF81aFbA2rba/openvqa/ckpts/ckpt_dataset_pt3_m59 -O ckpt_dataset_pt3_m59.zip
wget https://www.dropbox.com/sh/4xds64j6atmi68j/AAD3yhSB66NXPIHdUcfAu34Ra/openvqa/ckpts/ckpt_dataset_pt3_m61 -O ckpt_dataset_pt3_m61.zip
wget https://www.dropbox.com/sh/4xds64j6atmi68j/AADsd6s1KzBwR_2swo-3Raaxa/openvqa/ckpts/ckpt_dataset_pt3_m62 -O ckpt_dataset_pt3_m62.zip
wget https://www.dropbox.com/sh/4xds64j6atmi68j/AAAEfjQ-p58cPh555mITudV0a/openvqa/ckpts/ckpt_dataset_pt3_m63 -O ckpt_dataset_pt3_m63.zip
wget https://www.dropbox.com/sh/4xds64j6atmi68j/AABU_Mb9_Mt7dxz3Fnm1_9V0a/openvqa/ckpts/ckpt_dataset_pt3_m64 -O ckpt_dataset_pt3_m64.zip
wget https://www.dropbox.com/sh/4xds64j6atmi68j/AAA1TYredqJvDTWmFordaCL1a/openvqa/ckpts/ckpt_dataset_pt3_m65 -O ckpt_dataset_pt3_m65.zip
wget https://www.dropbox.com/sh/4xds64j6atmi68j/AADzPS1gMtqjm35x_rcNtOYda/openvqa/ckpts/ckpt_dataset_pt3_m66 -O ckpt_dataset_pt3_m66.zip
wget https://www.dropbox.com/sh/4xds64j6atmi68j/AAAk5cElHKN2lSWIHJeG89JPa/openvqa/ckpts/ckpt_dataset_pt3_m67 -O ckpt_dataset_pt3_m67.zip
wget https://www.dropbox.com/sh/4xds64j6atmi68j/AADxMcUZmbAu8tVs_YUsUi-Ca/openvqa/ckpts/ckpt_dataset_pt3_m68 -O ckpt_dataset_pt3_m68.zip
wget https://www.dropbox.com/sh/4xds64j6atmi68j/AACwU6PDawQ7LqEjdRTaBSnfa/openvqa/ckpts/ckpt_dataset_pt3_m69 -O ckpt_dataset_pt3_m69.zip
wget https://www.dropbox.com/sh/4xds64j6atmi68j/AACYAzW483K3tZFf9I9yAuGka/openvqa/ckpts/ckpt_dataset_pt3_m71 -O ckpt_dataset_pt3_m71.zip
wget https://www.dropbox.com/sh/4xds64j6atmi68j/AABA4NsMd2q1XjhsYtO3MURba/openvqa/ckpts/ckpt_dataset_pt3_m101 -O ckpt_dataset_pt3_m101.zip
wget https://www.dropbox.com/sh/4xds64j6atmi68j/AADglXDJ734smwjZUc1UsNbRa/openvqa/ckpts/ckpt_dataset_pt3_m102 -O ckpt_dataset_pt3_m102.zip
wget https://www.dropbox.com/sh/4xds64j6atmi68j/AACX78CxUj0EIhCuOGQIfn5Fa/openvqa/ckpts/ckpt_dataset_pt3_m103 -O ckpt_dataset_pt3_m103.zip
wget https://www.dropbox.com/sh/4xds64j6atmi68j/AACySdq1LbyceQN7ImZQV6oEa/openvqa/ckpts/ckpt_dataset_pt3_m104 -O ckpt_dataset_pt3_m104.zip
wget https://www.dropbox.com/sh/4xds64j6atmi68j/AAAUy8YcGp0A7obDHDgFoIaJa/openvqa/ckpts/ckpt_dataset_pt3_m105 -O ckpt_dataset_pt3_m105.zip
wget https://www.dropbox.com/sh/4xds64j6atmi68j/AAAymebKO91rAy4dUxP8zWUra/openvqa/ckpts/ckpt_dataset_pt3_m106 -O ckpt_dataset_pt3_m106.zip
wget https://www.dropbox.com/sh/4xds64j6atmi68j/AACw2jkF9-ndKtpvmLCNX0YXa/openvqa/ckpts/ckpt_dataset_pt3_m107 -O ckpt_dataset_pt3_m107.zip
wget https://www.dropbox.com/sh/4xds64j6atmi68j/AABeqyHyn1hTcKLTgCUt2aoma/openvqa/ckpts/ckpt_dataset_pt3_m108 -O ckpt_dataset_pt3_m108.zip
wget https://www.dropbox.com/sh/4xds64j6atmi68j/AABwsPCXspYAnHhrQBZnTsT4a/openvqa/ckpts/ckpt_dataset_pt3_m109 -O ckpt_dataset_pt3_m109.zip
wget https://www.dropbox.com/sh/4xds64j6atmi68j/AAAEZUB9dUdXJcIWxZBfnjmna/openvqa/ckpts/ckpt_dataset_pt3_m111 -O ckpt_dataset_pt3_m111.zip
wget https://www.dropbox.com/sh/4xds64j6atmi68j/AABk5W86oRMC_-oMxR0QhA72a/openvqa/ckpts/ckpt_dataset_pt3_m112 -O ckpt_dataset_pt3_m112.zip
wget https://www.dropbox.com/sh/4xds64j6atmi68j/AADRU9FrBFTDtLb_FYH9LME5a/openvqa/ckpts/ckpt_dataset_pt3_m113 -O ckpt_dataset_pt3_m113.zip
wget https://www.dropbox.com/sh/4xds64j6atmi68j/AADTYStHKoK6PJ_4NoXNHaAZa/openvqa/ckpts/ckpt_dataset_pt3_m114 -O ckpt_dataset_pt3_m114.zip
wget https://www.dropbox.com/sh/4xds64j6atmi68j/AAA52cMn9T67rvl3zo9a2rMVa/openvqa/ckpts/ckpt_dataset_pt3_m115 -O ckpt_dataset_pt3_m115.zip
wget https://www.dropbox.com/sh/4xds64j6atmi68j/AAA0bd3FF7cRCeTp3xY_JjHQa/openvqa/ckpts/ckpt_dataset_pt3_m116 -O ckpt_dataset_pt3_m116.zip
wget https://www.dropbox.com/sh/4xds64j6atmi68j/AADFaxhinVH6eS_80r21Bki6a/openvqa/ckpts/ckpt_dataset_pt3_m117 -O ckpt_dataset_pt3_m117.zip
wget https://www.dropbox.com/sh/4xds64j6atmi68j/AADTs4l9IZPT9CSF837ZjLwfa/openvqa/ckpts/ckpt_dataset_pt3_m118 -O ckpt_dataset_pt3_m118.zip
wget https://www.dropbox.com/sh/4xds64j6atmi68j/AAANt4Aj0jRyQnZtFEykBUIza/openvqa/ckpts/ckpt_dataset_pt3_m119 -O ckpt_dataset_pt3_m119.zip



unzip -n ckpt_dataset_pt1_m1.zip -d ckpt_dataset_pt1_m1
unzip -n ckpt_dataset_pt1_m2.zip -d ckpt_dataset_pt1_m2
unzip -n ckpt_dataset_pt1_m3.zip -d ckpt_dataset_pt1_m3
unzip -n ckpt_dataset_pt1_m4.zip -d ckpt_dataset_pt1_m4
unzip -n ckpt_dataset_pt1_m5.zip -d ckpt_dataset_pt1_m5
unzip -n ckpt_dataset_pt1_m6.zip -d ckpt_dataset_pt1_m6
unzip -n ckpt_dataset_pt1_m7.zip -d ckpt_dataset_pt1_m7
unzip -n ckpt_dataset_pt1_m8.zip -d ckpt_dataset_pt1_m8
unzip -n ckpt_dataset_pt1_m9.zip -d ckpt_dataset_pt1_m9
unzip -n ckpt_dataset_pt1_m11.zip -d ckpt_dataset_pt1_m11
unzip -n ckpt_dataset_pt1_m12.zip -d ckpt_dataset_pt1_m12
unzip -n ckpt_dataset_pt1_m13.zip -d ckpt_dataset_pt1_m13
unzip -n ckpt_dataset_pt1_m14.zip -d ckpt_dataset_pt1_m14
unzip -n ckpt_dataset_pt1_m15.zip -d ckpt_dataset_pt1_m15
unzip -n ckpt_dataset_pt1_m16.zip -d ckpt_dataset_pt1_m16
unzip -n ckpt_dataset_pt1_m17.zip -d ckpt_dataset_pt1_m17
unzip -n ckpt_dataset_pt1_m18.zip -d ckpt_dataset_pt1_m18
unzip -n ckpt_dataset_pt1_m19.zip -d ckpt_dataset_pt1_m19
unzip -n ckpt_dataset_pt1_m21.zip -d ckpt_dataset_pt1_m21
unzip -n ckpt_dataset_pt1_m22.zip -d ckpt_dataset_pt1_m22
unzip -n ckpt_dataset_pt1_m23.zip -d ckpt_dataset_pt1_m23
unzip -n ckpt_dataset_pt1_m24.zip -d ckpt_dataset_pt1_m24
unzip -n ckpt_dataset_pt1_m25.zip -d ckpt_dataset_pt1_m25
unzip -n ckpt_dataset_pt1_m26.zip -d ckpt_dataset_pt1_m26
unzip -n ckpt_dataset_pt1_m27.zip -d ckpt_dataset_pt1_m27
unzip -n ckpt_dataset_pt1_m28.zip -d ckpt_dataset_pt1_m28
unzip -n ckpt_dataset_pt1_m29.zip -d ckpt_dataset_pt1_m29
unzip -n ckpt_dataset_pt1_m31.zip -d ckpt_dataset_pt1_m31
unzip -n ckpt_dataset_pt1_m32.zip -d ckpt_dataset_pt1_m32
unzip -n ckpt_dataset_pt1_m33.zip -d ckpt_dataset_pt1_m33
unzip -n ckpt_dataset_pt1_m34.zip -d ckpt_dataset_pt1_m34
unzip -n ckpt_dataset_pt1_m35.zip -d ckpt_dataset_pt1_m35
unzip -n ckpt_dataset_pt1_m36.zip -d ckpt_dataset_pt1_m36
unzip -n ckpt_dataset_pt1_m37.zip -d ckpt_dataset_pt1_m37
unzip -n ckpt_dataset_pt1_m38.zip -d ckpt_dataset_pt1_m38
unzip -n ckpt_dataset_pt1_m39.zip -d ckpt_dataset_pt1_m39
unzip -n ckpt_dataset_pt1_m41.zip -d ckpt_dataset_pt1_m41
unzip -n ckpt_dataset_pt1_m42.zip -d ckpt_dataset_pt1_m42
unzip -n ckpt_dataset_pt1_m43.zip -d ckpt_dataset_pt1_m43
unzip -n ckpt_dataset_pt1_m44.zip -d ckpt_dataset_pt1_m44
unzip -n ckpt_dataset_pt1_m45.zip -d ckpt_dataset_pt1_m45
unzip -n ckpt_dataset_pt1_m46.zip -d ckpt_dataset_pt1_m46
unzip -n ckpt_dataset_pt1_m47.zip -d ckpt_dataset_pt1_m47
unzip -n ckpt_dataset_pt1_m48.zip -d ckpt_dataset_pt1_m48
unzip -n ckpt_dataset_pt1_m49.zip -d ckpt_dataset_pt1_m49
unzip -n ckpt_dataset_pt1_m51.zip -d ckpt_dataset_pt1_m51
unzip -n ckpt_dataset_pt1_m52.zip -d ckpt_dataset_pt1_m52
unzip -n ckpt_dataset_pt1_m53.zip -d ckpt_dataset_pt1_m53
unzip -n ckpt_dataset_pt1_m54.zip -d ckpt_dataset_pt1_m54
unzip -n ckpt_dataset_pt1_m55.zip -d ckpt_dataset_pt1_m55
unzip -n ckpt_dataset_pt1_m56.zip -d ckpt_dataset_pt1_m56
unzip -n ckpt_dataset_pt1_m57.zip -d ckpt_dataset_pt1_m57
unzip -n ckpt_dataset_pt1_m58.zip -d ckpt_dataset_pt1_m58
unzip -n ckpt_dataset_pt1_m59.zip -d ckpt_dataset_pt1_m59
unzip -n ckpt_dataset_pt1_m61.zip -d ckpt_dataset_pt1_m61
unzip -n ckpt_dataset_pt1_m62.zip -d ckpt_dataset_pt1_m62
unzip -n ckpt_dataset_pt1_m63.zip -d ckpt_dataset_pt1_m63
unzip -n ckpt_dataset_pt1_m64.zip -d ckpt_dataset_pt1_m64
unzip -n ckpt_dataset_pt1_m65.zip -d ckpt_dataset_pt1_m65
unzip -n ckpt_dataset_pt1_m66.zip -d ckpt_dataset_pt1_m66
unzip -n ckpt_dataset_pt1_m67.zip -d ckpt_dataset_pt1_m67
unzip -n ckpt_dataset_pt1_m68.zip -d ckpt_dataset_pt1_m68
unzip -n ckpt_dataset_pt1_m69.zip -d ckpt_dataset_pt1_m69
unzip -n ckpt_dataset_pt1_m71.zip -d ckpt_dataset_pt1_m71
unzip -n ckpt_dataset_pt1_m72.zip -d ckpt_dataset_pt1_m72
unzip -n ckpt_dataset_pt1_m73.zip -d ckpt_dataset_pt1_m73
unzip -n ckpt_dataset_pt1_m74.zip -d ckpt_dataset_pt1_m74
unzip -n ckpt_dataset_pt1_m75.zip -d ckpt_dataset_pt1_m75
unzip -n ckpt_dataset_pt1_m76.zip -d ckpt_dataset_pt1_m76
unzip -n ckpt_dataset_pt1_m77.zip -d ckpt_dataset_pt1_m77
unzip -n ckpt_dataset_pt1_m78.zip -d ckpt_dataset_pt1_m78
unzip -n ckpt_dataset_pt1_m79.zip -d ckpt_dataset_pt1_m79
unzip -n ckpt_dataset_pt1_m81.zip -d ckpt_dataset_pt1_m81
unzip -n ckpt_dataset_pt1_m82.zip -d ckpt_dataset_pt1_m82
unzip -n ckpt_dataset_pt1_m83.zip -d ckpt_dataset_pt1_m83
unzip -n ckpt_dataset_pt1_m84.zip -d ckpt_dataset_pt1_m84
unzip -n ckpt_dataset_pt1_m85.zip -d ckpt_dataset_pt1_m85
unzip -n ckpt_dataset_pt1_m86.zip -d ckpt_dataset_pt1_m86
unzip -n ckpt_dataset_pt1_m87.zip -d ckpt_dataset_pt1_m87
unzip -n ckpt_dataset_pt1_m88.zip -d ckpt_dataset_pt1_m88
unzip -n ckpt_dataset_pt1_m89.zip -d ckpt_dataset_pt1_m89
unzip -n ckpt_dataset_pt1_m91.zip -d ckpt_dataset_pt1_m91
unzip -n ckpt_dataset_pt1_m92.zip -d ckpt_dataset_pt1_m92
unzip -n ckpt_dataset_pt1_m93.zip -d ckpt_dataset_pt1_m93
unzip -n ckpt_dataset_pt1_m94.zip -d ckpt_dataset_pt1_m94
unzip -n ckpt_dataset_pt1_m95.zip -d ckpt_dataset_pt1_m95
unzip -n ckpt_dataset_pt1_m96.zip -d ckpt_dataset_pt1_m96
unzip -n ckpt_dataset_pt1_m97.zip -d ckpt_dataset_pt1_m97
unzip -n ckpt_dataset_pt1_m98.zip -d ckpt_dataset_pt1_m98
unzip -n ckpt_dataset_pt1_m99.zip -d ckpt_dataset_pt1_m99
unzip -n ckpt_dataset_pt1_m101.zip -d ckpt_dataset_pt1_m101
unzip -n ckpt_dataset_pt1_m102.zip -d ckpt_dataset_pt1_m102
unzip -n ckpt_dataset_pt1_m103.zip -d ckpt_dataset_pt1_m103
unzip -n ckpt_dataset_pt1_m104.zip -d ckpt_dataset_pt1_m104
unzip -n ckpt_dataset_pt1_m105.zip -d ckpt_dataset_pt1_m105
unzip -n ckpt_dataset_pt1_m106.zip -d ckpt_dataset_pt1_m106
unzip -n ckpt_dataset_pt1_m107.zip -d ckpt_dataset_pt1_m107
unzip -n ckpt_dataset_pt1_m108.zip -d ckpt_dataset_pt1_m108
unzip -n ckpt_dataset_pt1_m109.zip -d ckpt_dataset_pt1_m109
unzip -n ckpt_dataset_pt1_m111.zip -d ckpt_dataset_pt1_m111
unzip -n ckpt_dataset_pt1_m112.zip -d ckpt_dataset_pt1_m112
unzip -n ckpt_dataset_pt1_m113.zip -d ckpt_dataset_pt1_m113
unzip -n ckpt_dataset_pt1_m114.zip -d ckpt_dataset_pt1_m114
unzip -n ckpt_dataset_pt1_m115.zip -d ckpt_dataset_pt1_m115
unzip -n ckpt_dataset_pt1_m116.zip -d ckpt_dataset_pt1_m116
unzip -n ckpt_dataset_pt1_m117.zip -d ckpt_dataset_pt1_m117
unzip -n ckpt_dataset_pt1_m118.zip -d ckpt_dataset_pt1_m118
unzip -n ckpt_dataset_pt1_m119.zip -d ckpt_dataset_pt1_m119
unzip -n ckpt_dataset_pt1_m121.zip -d ckpt_dataset_pt1_m121
unzip -n ckpt_dataset_pt1_m122.zip -d ckpt_dataset_pt1_m122
unzip -n ckpt_dataset_pt1_m123.zip -d ckpt_dataset_pt1_m123
unzip -n ckpt_dataset_pt1_m124.zip -d ckpt_dataset_pt1_m124
unzip -n ckpt_dataset_pt1_m125.zip -d ckpt_dataset_pt1_m125
unzip -n ckpt_dataset_pt1_m126.zip -d ckpt_dataset_pt1_m126
unzip -n ckpt_dataset_pt1_m127.zip -d ckpt_dataset_pt1_m127
unzip -n ckpt_dataset_pt1_m128.zip -d ckpt_dataset_pt1_m128
unzip -n ckpt_dataset_pt1_m129.zip -d ckpt_dataset_pt1_m129
unzip -n ckpt_dataset_pt1_m131.zip -d ckpt_dataset_pt1_m131
unzip -n ckpt_dataset_pt1_m132.zip -d ckpt_dataset_pt1_m132
unzip -n ckpt_dataset_pt1_m133.zip -d ckpt_dataset_pt1_m133
unzip -n ckpt_dataset_pt1_m134.zip -d ckpt_dataset_pt1_m134
unzip -n ckpt_dataset_pt1_m135.zip -d ckpt_dataset_pt1_m135
unzip -n ckpt_dataset_pt1_m136.zip -d ckpt_dataset_pt1_m136
unzip -n ckpt_dataset_pt1_m137.zip -d ckpt_dataset_pt1_m137
unzip -n ckpt_dataset_pt1_m138.zip -d ckpt_dataset_pt1_m138
unzip -n ckpt_dataset_pt1_m139.zip -d ckpt_dataset_pt1_m139
unzip -n ckpt_dataset_pt1_m141.zip -d ckpt_dataset_pt1_m141
unzip -n ckpt_dataset_pt1_m142.zip -d ckpt_dataset_pt1_m142
unzip -n ckpt_dataset_pt1_m143.zip -d ckpt_dataset_pt1_m143
unzip -n ckpt_dataset_pt1_m144.zip -d ckpt_dataset_pt1_m144
unzip -n ckpt_dataset_pt1_m145.zip -d ckpt_dataset_pt1_m145
unzip -n ckpt_dataset_pt1_m146.zip -d ckpt_dataset_pt1_m146
unzip -n ckpt_dataset_pt1_m147.zip -d ckpt_dataset_pt1_m147
unzip -n ckpt_dataset_pt1_m148.zip -d ckpt_dataset_pt1_m148
unzip -n ckpt_dataset_pt1_m149.zip -d ckpt_dataset_pt1_m149
unzip -n ckpt_dataset_pt1_m151.zip -d ckpt_dataset_pt1_m151
unzip -n ckpt_dataset_pt1_m152.zip -d ckpt_dataset_pt1_m152
unzip -n ckpt_dataset_pt1_m153.zip -d ckpt_dataset_pt1_m153
unzip -n ckpt_dataset_pt1_m154.zip -d ckpt_dataset_pt1_m154
unzip -n ckpt_dataset_pt1_m155.zip -d ckpt_dataset_pt1_m155
unzip -n ckpt_dataset_pt1_m156.zip -d ckpt_dataset_pt1_m156
unzip -n ckpt_dataset_pt1_m157.zip -d ckpt_dataset_pt1_m157
unzip -n ckpt_dataset_pt1_m158.zip -d ckpt_dataset_pt1_m158
unzip -n ckpt_dataset_pt1_m159.zip -d ckpt_dataset_pt1_m159
unzip -n ckpt_dataset_pt1_m161.zip -d ckpt_dataset_pt1_m161
unzip -n ckpt_dataset_pt1_m162.zip -d ckpt_dataset_pt1_m162
unzip -n ckpt_dataset_pt1_m163.zip -d ckpt_dataset_pt1_m163
unzip -n ckpt_dataset_pt1_m164.zip -d ckpt_dataset_pt1_m164
unzip -n ckpt_dataset_pt1_m165.zip -d ckpt_dataset_pt1_m165
unzip -n ckpt_dataset_pt1_m166.zip -d ckpt_dataset_pt1_m166
unzip -n ckpt_dataset_pt1_m167.zip -d ckpt_dataset_pt1_m167
unzip -n ckpt_dataset_pt1_m168.zip -d ckpt_dataset_pt1_m168
unzip -n ckpt_dataset_pt1_m169.zip -d ckpt_dataset_pt1_m169
unzip -n ckpt_dataset_pt1_m171.zip -d ckpt_dataset_pt1_m171
unzip -n ckpt_dataset_pt1_m172.zip -d ckpt_dataset_pt1_m172
unzip -n ckpt_dataset_pt1_m173.zip -d ckpt_dataset_pt1_m173
unzip -n ckpt_dataset_pt1_m174.zip -d ckpt_dataset_pt1_m174
unzip -n ckpt_dataset_pt1_m175.zip -d ckpt_dataset_pt1_m175
unzip -n ckpt_dataset_pt1_m176.zip -d ckpt_dataset_pt1_m176
unzip -n ckpt_dataset_pt1_m177.zip -d ckpt_dataset_pt1_m177
unzip -n ckpt_dataset_pt1_m178.zip -d ckpt_dataset_pt1_m178
unzip -n ckpt_dataset_pt1_m179.zip -d ckpt_dataset_pt1_m179
unzip -n ckpt_dataset_pt1_m181.zip -d ckpt_dataset_pt1_m181
unzip -n ckpt_dataset_pt1_m182.zip -d ckpt_dataset_pt1_m182
unzip -n ckpt_dataset_pt1_m183.zip -d ckpt_dataset_pt1_m183
unzip -n ckpt_dataset_pt1_m184.zip -d ckpt_dataset_pt1_m184
unzip -n ckpt_dataset_pt1_m185.zip -d ckpt_dataset_pt1_m185
unzip -n ckpt_dataset_pt1_m186.zip -d ckpt_dataset_pt1_m186
unzip -n ckpt_dataset_pt1_m187.zip -d ckpt_dataset_pt1_m187
unzip -n ckpt_dataset_pt1_m188.zip -d ckpt_dataset_pt1_m188
unzip -n ckpt_dataset_pt1_m189.zip -d ckpt_dataset_pt1_m189
unzip -n ckpt_dataset_pt1_m191.zip -d ckpt_dataset_pt1_m191
unzip -n ckpt_dataset_pt1_m192.zip -d ckpt_dataset_pt1_m192
unzip -n ckpt_dataset_pt1_m193.zip -d ckpt_dataset_pt1_m193
unzip -n ckpt_dataset_pt1_m194.zip -d ckpt_dataset_pt1_m194
unzip -n ckpt_dataset_pt1_m195.zip -d ckpt_dataset_pt1_m195
unzip -n ckpt_dataset_pt1_m196.zip -d ckpt_dataset_pt1_m196
unzip -n ckpt_dataset_pt1_m197.zip -d ckpt_dataset_pt1_m197
unzip -n ckpt_dataset_pt1_m198.zip -d ckpt_dataset_pt1_m198
unzip -n ckpt_dataset_pt1_m199.zip -d ckpt_dataset_pt1_m199
unzip -n ckpt_dataset_pt1_m201.zip -d ckpt_dataset_pt1_m201
unzip -n ckpt_dataset_pt1_m202.zip -d ckpt_dataset_pt1_m202
unzip -n ckpt_dataset_pt1_m203.zip -d ckpt_dataset_pt1_m203
unzip -n ckpt_dataset_pt1_m204.zip -d ckpt_dataset_pt1_m204
unzip -n ckpt_dataset_pt1_m205.zip -d ckpt_dataset_pt1_m205
unzip -n ckpt_dataset_pt1_m206.zip -d ckpt_dataset_pt1_m206
unzip -n ckpt_dataset_pt1_m207.zip -d ckpt_dataset_pt1_m207
unzip -n ckpt_dataset_pt1_m208.zip -d ckpt_dataset_pt1_m208
unzip -n ckpt_dataset_pt1_m209.zip -d ckpt_dataset_pt1_m209
unzip -n ckpt_dataset_pt1_m211.zip -d ckpt_dataset_pt1_m211
unzip -n ckpt_dataset_pt1_m212.zip -d ckpt_dataset_pt1_m212
unzip -n ckpt_dataset_pt1_m213.zip -d ckpt_dataset_pt1_m213
unzip -n ckpt_dataset_pt1_m214.zip -d ckpt_dataset_pt1_m214
unzip -n ckpt_dataset_pt1_m215.zip -d ckpt_dataset_pt1_m215
unzip -n ckpt_dataset_pt1_m216.zip -d ckpt_dataset_pt1_m216
unzip -n ckpt_dataset_pt1_m217.zip -d ckpt_dataset_pt1_m217
unzip -n ckpt_dataset_pt1_m218.zip -d ckpt_dataset_pt1_m218
unzip -n ckpt_dataset_pt1_m219.zip -d ckpt_dataset_pt1_m219
unzip -n ckpt_dataset_pt1_m221.zip -d ckpt_dataset_pt1_m221
unzip -n ckpt_dataset_pt1_m222.zip -d ckpt_dataset_pt1_m222
unzip -n ckpt_dataset_pt1_m223.zip -d ckpt_dataset_pt1_m223
unzip -n ckpt_dataset_pt1_m224.zip -d ckpt_dataset_pt1_m224
unzip -n ckpt_dataset_pt1_m225.zip -d ckpt_dataset_pt1_m225
unzip -n ckpt_dataset_pt1_m226.zip -d ckpt_dataset_pt1_m226
unzip -n ckpt_dataset_pt1_m227.zip -d ckpt_dataset_pt1_m227
unzip -n ckpt_dataset_pt1_m228.zip -d ckpt_dataset_pt1_m228
unzip -n ckpt_dataset_pt1_m229.zip -d ckpt_dataset_pt1_m229
unzip -n ckpt_dataset_pt1_m231.zip -d ckpt_dataset_pt1_m231
unzip -n ckpt_dataset_pt1_m232.zip -d ckpt_dataset_pt1_m232
unzip -n ckpt_dataset_pt1_m233.zip -d ckpt_dataset_pt1_m233
unzip -n ckpt_dataset_pt1_m234.zip -d ckpt_dataset_pt1_m234
unzip -n ckpt_dataset_pt1_m235.zip -d ckpt_dataset_pt1_m235
unzip -n ckpt_dataset_pt1_m236.zip -d ckpt_dataset_pt1_m236
unzip -n ckpt_dataset_pt1_m237.zip -d ckpt_dataset_pt1_m237
unzip -n ckpt_dataset_pt1_m238.zip -d ckpt_dataset_pt1_m238
unzip -n ckpt_dataset_pt1_m239.zip -d ckpt_dataset_pt1_m239
unzip -n ckpt_dataset_pt2_m1.zip -d ckpt_dataset_pt2_m1
unzip -n ckpt_dataset_pt2_m2.zip -d ckpt_dataset_pt2_m2
unzip -n ckpt_dataset_pt2_m3.zip -d ckpt_dataset_pt2_m3
unzip -n ckpt_dataset_pt2_m4.zip -d ckpt_dataset_pt2_m4
unzip -n ckpt_dataset_pt2_m5.zip -d ckpt_dataset_pt2_m5
unzip -n ckpt_dataset_pt2_m6.zip -d ckpt_dataset_pt2_m6
unzip -n ckpt_dataset_pt2_m7.zip -d ckpt_dataset_pt2_m7
unzip -n ckpt_dataset_pt2_m8.zip -d ckpt_dataset_pt2_m8
unzip -n ckpt_dataset_pt2_m9.zip -d ckpt_dataset_pt2_m9
unzip -n ckpt_dataset_pt2_m11.zip -d ckpt_dataset_pt2_m11
unzip -n ckpt_dataset_pt2_m12.zip -d ckpt_dataset_pt2_m12
unzip -n ckpt_dataset_pt2_m13.zip -d ckpt_dataset_pt2_m13
unzip -n ckpt_dataset_pt2_m14.zip -d ckpt_dataset_pt2_m14
unzip -n ckpt_dataset_pt2_m15.zip -d ckpt_dataset_pt2_m15
unzip -n ckpt_dataset_pt2_m16.zip -d ckpt_dataset_pt2_m16
unzip -n ckpt_dataset_pt2_m17.zip -d ckpt_dataset_pt2_m17
unzip -n ckpt_dataset_pt2_m18.zip -d ckpt_dataset_pt2_m18
unzip -n ckpt_dataset_pt2_m19.zip -d ckpt_dataset_pt2_m19
unzip -n ckpt_dataset_pt2_m21.zip -d ckpt_dataset_pt2_m21
unzip -n ckpt_dataset_pt2_m22.zip -d ckpt_dataset_pt2_m22
unzip -n ckpt_dataset_pt2_m23.zip -d ckpt_dataset_pt2_m23
unzip -n ckpt_dataset_pt2_m24.zip -d ckpt_dataset_pt2_m24
unzip -n ckpt_dataset_pt2_m25.zip -d ckpt_dataset_pt2_m25
unzip -n ckpt_dataset_pt2_m26.zip -d ckpt_dataset_pt2_m26
unzip -n ckpt_dataset_pt2_m27.zip -d ckpt_dataset_pt2_m27
unzip -n ckpt_dataset_pt2_m28.zip -d ckpt_dataset_pt2_m28
unzip -n ckpt_dataset_pt2_m29.zip -d ckpt_dataset_pt2_m29
unzip -n ckpt_dataset_pt2_m31.zip -d ckpt_dataset_pt2_m31
unzip -n ckpt_dataset_pt2_m32.zip -d ckpt_dataset_pt2_m32
unzip -n ckpt_dataset_pt2_m33.zip -d ckpt_dataset_pt2_m33
unzip -n ckpt_dataset_pt2_m34.zip -d ckpt_dataset_pt2_m34
unzip -n ckpt_dataset_pt2_m35.zip -d ckpt_dataset_pt2_m35
unzip -n ckpt_dataset_pt2_m36.zip -d ckpt_dataset_pt2_m36
unzip -n ckpt_dataset_pt2_m37.zip -d ckpt_dataset_pt2_m37
unzip -n ckpt_dataset_pt2_m38.zip -d ckpt_dataset_pt2_m38
unzip -n ckpt_dataset_pt2_m39.zip -d ckpt_dataset_pt2_m39
unzip -n ckpt_dataset_pt2_m41.zip -d ckpt_dataset_pt2_m41
unzip -n ckpt_dataset_pt2_m42.zip -d ckpt_dataset_pt2_m42
unzip -n ckpt_dataset_pt2_m43.zip -d ckpt_dataset_pt2_m43
unzip -n ckpt_dataset_pt2_m44.zip -d ckpt_dataset_pt2_m44
unzip -n ckpt_dataset_pt2_m45.zip -d ckpt_dataset_pt2_m45
unzip -n ckpt_dataset_pt2_m46.zip -d ckpt_dataset_pt2_m46
unzip -n ckpt_dataset_pt2_m47.zip -d ckpt_dataset_pt2_m47
unzip -n ckpt_dataset_pt2_m48.zip -d ckpt_dataset_pt2_m48
unzip -n ckpt_dataset_pt2_m49.zip -d ckpt_dataset_pt2_m49
unzip -n ckpt_dataset_pt2_m51.zip -d ckpt_dataset_pt2_m51
unzip -n ckpt_dataset_pt2_m52.zip -d ckpt_dataset_pt2_m52
unzip -n ckpt_dataset_pt2_m53.zip -d ckpt_dataset_pt2_m53
unzip -n ckpt_dataset_pt2_m54.zip -d ckpt_dataset_pt2_m54
unzip -n ckpt_dataset_pt2_m55.zip -d ckpt_dataset_pt2_m55
unzip -n ckpt_dataset_pt2_m56.zip -d ckpt_dataset_pt2_m56
unzip -n ckpt_dataset_pt2_m57.zip -d ckpt_dataset_pt2_m57
unzip -n ckpt_dataset_pt2_m58.zip -d ckpt_dataset_pt2_m58
unzip -n ckpt_dataset_pt2_m59.zip -d ckpt_dataset_pt2_m59
unzip -n ckpt_dataset_pt2_m61.zip -d ckpt_dataset_pt2_m61
unzip -n ckpt_dataset_pt2_m62.zip -d ckpt_dataset_pt2_m62
unzip -n ckpt_dataset_pt2_m63.zip -d ckpt_dataset_pt2_m63
unzip -n ckpt_dataset_pt2_m64.zip -d ckpt_dataset_pt2_m64
unzip -n ckpt_dataset_pt2_m65.zip -d ckpt_dataset_pt2_m65
unzip -n ckpt_dataset_pt2_m66.zip -d ckpt_dataset_pt2_m66
unzip -n ckpt_dataset_pt2_m67.zip -d ckpt_dataset_pt2_m67
unzip -n ckpt_dataset_pt2_m68.zip -d ckpt_dataset_pt2_m68
unzip -n ckpt_dataset_pt2_m69.zip -d ckpt_dataset_pt2_m69
unzip -n ckpt_dataset_pt2_m71.zip -d ckpt_dataset_pt2_m71
unzip -n ckpt_dataset_pt2_m72.zip -d ckpt_dataset_pt2_m72
unzip -n ckpt_dataset_pt2_m73.zip -d ckpt_dataset_pt2_m73
unzip -n ckpt_dataset_pt2_m74.zip -d ckpt_dataset_pt2_m74
unzip -n ckpt_dataset_pt2_m75.zip -d ckpt_dataset_pt2_m75
unzip -n ckpt_dataset_pt2_m76.zip -d ckpt_dataset_pt2_m76
unzip -n ckpt_dataset_pt2_m77.zip -d ckpt_dataset_pt2_m77
unzip -n ckpt_dataset_pt2_m78.zip -d ckpt_dataset_pt2_m78
unzip -n ckpt_dataset_pt2_m79.zip -d ckpt_dataset_pt2_m79
unzip -n ckpt_dataset_pt2_m81.zip -d ckpt_dataset_pt2_m81
unzip -n ckpt_dataset_pt2_m82.zip -d ckpt_dataset_pt2_m82
unzip -n ckpt_dataset_pt2_m83.zip -d ckpt_dataset_pt2_m83
unzip -n ckpt_dataset_pt2_m84.zip -d ckpt_dataset_pt2_m84
unzip -n ckpt_dataset_pt2_m85.zip -d ckpt_dataset_pt2_m85
unzip -n ckpt_dataset_pt2_m86.zip -d ckpt_dataset_pt2_m86
unzip -n ckpt_dataset_pt2_m87.zip -d ckpt_dataset_pt2_m87
unzip -n ckpt_dataset_pt2_m88.zip -d ckpt_dataset_pt2_m88
unzip -n ckpt_dataset_pt2_m89.zip -d ckpt_dataset_pt2_m89
unzip -n ckpt_dataset_pt2_m91.zip -d ckpt_dataset_pt2_m91
unzip -n ckpt_dataset_pt2_m92.zip -d ckpt_dataset_pt2_m92
unzip -n ckpt_dataset_pt2_m93.zip -d ckpt_dataset_pt2_m93
unzip -n ckpt_dataset_pt2_m94.zip -d ckpt_dataset_pt2_m94
unzip -n ckpt_dataset_pt2_m95.zip -d ckpt_dataset_pt2_m95
unzip -n ckpt_dataset_pt2_m96.zip -d ckpt_dataset_pt2_m96
unzip -n ckpt_dataset_pt2_m97.zip -d ckpt_dataset_pt2_m97
unzip -n ckpt_dataset_pt2_m98.zip -d ckpt_dataset_pt2_m98
unzip -n ckpt_dataset_pt2_m99.zip -d ckpt_dataset_pt2_m99
unzip -n ckpt_dataset_pt2_m101.zip -d ckpt_dataset_pt2_m101
unzip -n ckpt_dataset_pt2_m102.zip -d ckpt_dataset_pt2_m102
unzip -n ckpt_dataset_pt2_m103.zip -d ckpt_dataset_pt2_m103
unzip -n ckpt_dataset_pt2_m104.zip -d ckpt_dataset_pt2_m104
unzip -n ckpt_dataset_pt2_m105.zip -d ckpt_dataset_pt2_m105
unzip -n ckpt_dataset_pt2_m106.zip -d ckpt_dataset_pt2_m106
unzip -n ckpt_dataset_pt2_m107.zip -d ckpt_dataset_pt2_m107
unzip -n ckpt_dataset_pt2_m108.zip -d ckpt_dataset_pt2_m108
unzip -n ckpt_dataset_pt2_m109.zip -d ckpt_dataset_pt2_m109
unzip -n ckpt_dataset_pt2_m111.zip -d ckpt_dataset_pt2_m111
unzip -n ckpt_dataset_pt2_m112.zip -d ckpt_dataset_pt2_m112
unzip -n ckpt_dataset_pt2_m113.zip -d ckpt_dataset_pt2_m113
unzip -n ckpt_dataset_pt2_m114.zip -d ckpt_dataset_pt2_m114
unzip -n ckpt_dataset_pt2_m115.zip -d ckpt_dataset_pt2_m115
unzip -n ckpt_dataset_pt2_m116.zip -d ckpt_dataset_pt2_m116
unzip -n ckpt_dataset_pt2_m117.zip -d ckpt_dataset_pt2_m117
unzip -n ckpt_dataset_pt2_m118.zip -d ckpt_dataset_pt2_m118
unzip -n ckpt_dataset_pt2_m119.zip -d ckpt_dataset_pt2_m119
unzip -n ckpt_dataset_pt3_m1.zip -d ckpt_dataset_pt3_m1
unzip -n ckpt_dataset_pt3_m2.zip -d ckpt_dataset_pt3_m2
unzip -n ckpt_dataset_pt3_m3.zip -d ckpt_dataset_pt3_m3
unzip -n ckpt_dataset_pt3_m4.zip -d ckpt_dataset_pt3_m4
unzip -n ckpt_dataset_pt3_m5.zip -d ckpt_dataset_pt3_m5
unzip -n ckpt_dataset_pt3_m6.zip -d ckpt_dataset_pt3_m6
unzip -n ckpt_dataset_pt3_m7.zip -d ckpt_dataset_pt3_m7
unzip -n ckpt_dataset_pt3_m11.zip -d ckpt_dataset_pt3_m11
unzip -n ckpt_dataset_pt3_m12.zip -d ckpt_dataset_pt3_m12
unzip -n ckpt_dataset_pt3_m13.zip -d ckpt_dataset_pt3_m13
unzip -n ckpt_dataset_pt3_m14.zip -d ckpt_dataset_pt3_m14
unzip -n ckpt_dataset_pt3_m15.zip -d ckpt_dataset_pt3_m15
unzip -n ckpt_dataset_pt3_m16.zip -d ckpt_dataset_pt3_m16
unzip -n ckpt_dataset_pt3_m17.zip -d ckpt_dataset_pt3_m17
unzip -n ckpt_dataset_pt3_m18.zip -d ckpt_dataset_pt3_m18
unzip -n ckpt_dataset_pt3_m19.zip -d ckpt_dataset_pt3_m19
unzip -n ckpt_dataset_pt3_m21.zip -d ckpt_dataset_pt3_m21
unzip -n ckpt_dataset_pt3_m22.zip -d ckpt_dataset_pt3_m22
unzip -n ckpt_dataset_pt3_m23.zip -d ckpt_dataset_pt3_m23
unzip -n ckpt_dataset_pt3_m24.zip -d ckpt_dataset_pt3_m24
unzip -n ckpt_dataset_pt3_m25.zip -d ckpt_dataset_pt3_m25
unzip -n ckpt_dataset_pt3_m26.zip -d ckpt_dataset_pt3_m26
unzip -n ckpt_dataset_pt3_m27.zip -d ckpt_dataset_pt3_m27
unzip -n ckpt_dataset_pt3_m28.zip -d ckpt_dataset_pt3_m28
unzip -n ckpt_dataset_pt3_m29.zip -d ckpt_dataset_pt3_m29
unzip -n ckpt_dataset_pt3_m31.zip -d ckpt_dataset_pt3_m31
unzip -n ckpt_dataset_pt3_m32.zip -d ckpt_dataset_pt3_m32
unzip -n ckpt_dataset_pt3_m33.zip -d ckpt_dataset_pt3_m33
unzip -n ckpt_dataset_pt3_m34.zip -d ckpt_dataset_pt3_m34
unzip -n ckpt_dataset_pt3_m35.zip -d ckpt_dataset_pt3_m35
unzip -n ckpt_dataset_pt3_m36.zip -d ckpt_dataset_pt3_m36
unzip -n ckpt_dataset_pt3_m37.zip -d ckpt_dataset_pt3_m37
unzip -n ckpt_dataset_pt3_m38.zip -d ckpt_dataset_pt3_m38
unzip -n ckpt_dataset_pt3_m39.zip -d ckpt_dataset_pt3_m39
unzip -n ckpt_dataset_pt3_m41.zip -d ckpt_dataset_pt3_m41
unzip -n ckpt_dataset_pt3_m42.zip -d ckpt_dataset_pt3_m42
unzip -n ckpt_dataset_pt3_m43.zip -d ckpt_dataset_pt3_m43
unzip -n ckpt_dataset_pt3_m44.zip -d ckpt_dataset_pt3_m44
unzip -n ckpt_dataset_pt3_m45.zip -d ckpt_dataset_pt3_m45
unzip -n ckpt_dataset_pt3_m46.zip -d ckpt_dataset_pt3_m46
unzip -n ckpt_dataset_pt3_m47.zip -d ckpt_dataset_pt3_m47
unzip -n ckpt_dataset_pt3_m48.zip -d ckpt_dataset_pt3_m48
unzip -n ckpt_dataset_pt3_m49.zip -d ckpt_dataset_pt3_m49
unzip -n ckpt_dataset_pt3_m51.zip -d ckpt_dataset_pt3_m51
unzip -n ckpt_dataset_pt3_m52.zip -d ckpt_dataset_pt3_m52
unzip -n ckpt_dataset_pt3_m53.zip -d ckpt_dataset_pt3_m53
unzip -n ckpt_dataset_pt3_m54.zip -d ckpt_dataset_pt3_m54
unzip -n ckpt_dataset_pt3_m55.zip -d ckpt_dataset_pt3_m55
unzip -n ckpt_dataset_pt3_m56.zip -d ckpt_dataset_pt3_m56
unzip -n ckpt_dataset_pt3_m57.zip -d ckpt_dataset_pt3_m57
unzip -n ckpt_dataset_pt3_m58.zip -d ckpt_dataset_pt3_m58
unzip -n ckpt_dataset_pt3_m59.zip -d ckpt_dataset_pt3_m59
unzip -n ckpt_dataset_pt3_m61.zip -d ckpt_dataset_pt3_m61
unzip -n ckpt_dataset_pt3_m62.zip -d ckpt_dataset_pt3_m62
unzip -n ckpt_dataset_pt3_m63.zip -d ckpt_dataset_pt3_m63
unzip -n ckpt_dataset_pt3_m64.zip -d ckpt_dataset_pt3_m64
unzip -n ckpt_dataset_pt3_m65.zip -d ckpt_dataset_pt3_m65
unzip -n ckpt_dataset_pt3_m66.zip -d ckpt_dataset_pt3_m66
unzip -n ckpt_dataset_pt3_m67.zip -d ckpt_dataset_pt3_m67
unzip -n ckpt_dataset_pt3_m68.zip -d ckpt_dataset_pt3_m68
unzip -n ckpt_dataset_pt3_m69.zip -d ckpt_dataset_pt3_m69
unzip -n ckpt_dataset_pt3_m71.zip -d ckpt_dataset_pt3_m71
unzip -n ckpt_dataset_pt3_m101.zip -d ckpt_dataset_pt3_m101
unzip -n ckpt_dataset_pt3_m102.zip -d ckpt_dataset_pt3_m102
unzip -n ckpt_dataset_pt3_m103.zip -d ckpt_dataset_pt3_m103
unzip -n ckpt_dataset_pt3_m104.zip -d ckpt_dataset_pt3_m104
unzip -n ckpt_dataset_pt3_m105.zip -d ckpt_dataset_pt3_m105
unzip -n ckpt_dataset_pt3_m106.zip -d ckpt_dataset_pt3_m106
unzip -n ckpt_dataset_pt3_m107.zip -d ckpt_dataset_pt3_m107
unzip -n ckpt_dataset_pt3_m108.zip -d ckpt_dataset_pt3_m108
unzip -n ckpt_dataset_pt3_m109.zip -d ckpt_dataset_pt3_m109
unzip -n ckpt_dataset_pt3_m111.zip -d ckpt_dataset_pt3_m111
unzip -n ckpt_dataset_pt3_m112.zip -d ckpt_dataset_pt3_m112
unzip -n ckpt_dataset_pt3_m113.zip -d ckpt_dataset_pt3_m113
unzip -n ckpt_dataset_pt3_m114.zip -d ckpt_dataset_pt3_m114
unzip -n ckpt_dataset_pt3_m115.zip -d ckpt_dataset_pt3_m115
unzip -n ckpt_dataset_pt3_m116.zip -d ckpt_dataset_pt3_m116
unzip -n ckpt_dataset_pt3_m117.zip -d ckpt_dataset_pt3_m117
unzip -n ckpt_dataset_pt3_m118.zip -d ckpt_dataset_pt3_m118
unzip -n ckpt_dataset_pt3_m119.zip -d ckpt_dataset_pt3_m119

# rm *.zip


cd /data/TrojVQA/
mkdir -p  model_sets/v1
mv bottom-up-attention-vqa model_sets/v1
mv openvqa/ model_sets/v1/
