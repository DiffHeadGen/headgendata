如果我写的路径有错别字，请纠正

# 视频处理

视频文件路径：data/rebuttal/output

任务，按照要求处理文件夹中的视频

定义变量

data_dir = data/rebuttal

ori_video_dir = data/rebuttal/output

retarget_dir = data/rebuttal/retarget

定义方法

- extract_imgs: 使用 ffmpeg 命令，将 .mp4 文件提取图片到指定路径，按照 %d06.jpg 格式命名

## 视频格式

每个一个视频是宽高比 3:1 的视频（视频可能是 512*512 的三个视频拼接起来的，这个需要你自己判断）

然后现在需要使用 ffmpeg 命令来处理视频

- 裁剪视频为三个1:1 的视频，横向依次编号为1,2,3
    - 1是一张静态的图片，只需要保留第一帧，输出到 {retarget_dir}/image.jpg
    - 2 是source视频，输出到 {retarget_dir}/source.mp4，然后调用 extract_imgs 输出到 {retarget_dir}/source_img/
    - 3 是result视频，输出到 {retarget_dir}/result.mp4，然后调用 extract_imgs 输出到 {retarget_dir}/result_img/

构造一个class，有如下功能

- {data_dir} 中的 {ori_video_dir}每个视频文件作为构造函数输入，然后class有默认变量，{data_dir}
    - class 有这些 property
        - source_img_dir
        - result_img_dir
        - ori_video_dir

----
视频的实际比例不符合预期，我仔细检查了一下

视频的大小一般是：1552*528

然后实际每个部分是：512*520，每个部分都有 4pix 的边框

- 1552=4+512+4+512+4，528=4+520+4

现在的要求是，把 512*520 不带黑框的部分裁出来作为之前说的1,2,3部分，然后需要resize到512*512