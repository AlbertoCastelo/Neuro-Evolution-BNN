import uuid
import s3fs
import pandas as pd
import pyarrow.parquet as pq
import pyarrow
import os


def save_df_to_csv_in_s3(df, bucket, filename):
    s3 = _s3fs()
    with s3.open(F's3://{bucket}/{filename}', 'w') as f:
        df.to_csv(f, index=False)


def read_df_to_csv_in_s3(bucket, filename):
    s3 = _s3fs()
    with s3.open(F's3://{bucket}/{filename}', 'r') as f:
        df = pd.read_csv(f)
    return df


def save_df_to_parquet_in_s3(df, bucket, data_path, name='', chunks_size=100000):
    result = []
    s3 = _s3fs()
    chunks = _get_chunks(df, chunks_size)
    url = f's3://{bucket}/{data_path}'
    for chunk in chunks:
        uuid_ = uuid.uuid4()
        # TODO if suffix is empty we shouldn't add it
        filepath = f'{data_path}/{uuid_}-{name}.parquet.gzip'
        fullpath = f's3://{bucket}/{filepath}'
        with s3.open(fullpath, 'wb') as f:
            table = pyarrow.Table.from_pandas(chunk)
            pyarrow.parquet.write_table(table, f, compression='gzip', coerce_timestamps='ms')
            result.append(filepath)
    return result


def _get_chunks(df, chunks_size):
    if df.empty:
        return [df]
    return [df[i:i + chunks_size] for i in range(0, len(df), chunks_size)]


def read_df_from_parquet_in_s3(bucket, data_path):
    s3 = _s3fs()
    url = f's3://{bucket}/{data_path}'
    return pq.ParquetDataset(url, filesystem=s3).read_pandas().to_pandas()


def get_files_in_directory_s3(bucket, data_path):
    s3 = _s3fs()
    url = f's3://{bucket}/{data_path}'
    return s3.ls(url)


def delete_file_in_s3(bucket, filename):
    s3 = _s3fs()
    url = f's3://{bucket}/{filename}'
    s3.rm(url)


def move_files_in_s3(bucket, old_path, new_path):
    s3 = _s3fs()
    url_old = f's3://{bucket}/{old_path}'
    url_new = f's3://{bucket}/{new_path}'
    files = s3.ls(url_old)
    for file in files:
        s3.mv(file, url_new)


def _s3fs():
    client_kwargs = None
    if 'AWS_S3_HOST' in os.environ:
        client_kwargs = {'endpoint_url': os.environ['AWS_S3_HOST']}
    return s3fs.S3FileSystem(client_kwargs=client_kwargs)