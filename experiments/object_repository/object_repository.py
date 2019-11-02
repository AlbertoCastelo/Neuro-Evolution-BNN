import os
import tempfile
from contextlib import contextmanager
from urllib.parse import urlparse
import boto3
import botocore
DEFAULT_REGION = "eu-west-1"


class BucketCreationError(Exception):
    def __init__(self, message):
        self.message = message
        super(BucketCreationError, self).__init__(("{}").format(message))


class KeyNotFoundError(Exception):
    def __init__(self, message):
        self.message = message
        super(KeyNotFoundError, self).__init__(("{}").format(message))


class ObjectRepository:
    """ AWS S3 repository
    Relevant references:
        - http://boto3.readthedocs.io/en/latest/reference/core/session.html
        - https://botocore.readthedocs.io/en/stable/reference/config.html
        - https://boto3.readthedocs.io/en/latest/reference/services/s3.html#S3.Bucket.Object
        - https://boto3.readthedocs.io/en/latest/reference/services/s3.html#S3.Object
        - https://botocore.readthedocs.io/en/latest/reference/response.html#botocore.response.StreamingBody
    """
    PUBLIC_READ = 'public-read'

    def __init__(self, bucket, logger, region=None):
        self.bucket_name = bucket
        self.logger = logger
        self.region = region or os.environ.get("AWS_REGION", DEFAULT_REGION)
        self._s3_conn = None

    @property
    def s3_conn(self):
        if self._s3_conn is None:
            # TODO Pending usage of OrdinaryCallingFormat
            if os.environ.get('AWS_S3_HOST'):
                self._s3_conn = boto3.resource(
                    's3',
                    region_name=self.region,
                    endpoint_url=os.environ.get('AWS_S3_HOST'))
            else:
                self._s3_conn = boto3.resource(
                    's3',
                    region_name=self.region)
        return self._s3_conn

    def set_from_file(self, key, data_path, policy=None, content_type=None, meta_data=None, **kwargs):
        with self._operate_with_new_key(key) as s3_key:
            params = self._prepare_params_for_set(policy, content_type, meta_data)
            with open(data_path, 'rb') as f:
                s3_key.put(Body=f, **params, **kwargs)
            self.logger.debug(f'KeyUploaded: {key}')

    def set(self, key, content, policy=None, content_type=None, meta_data=None):
        if content is None:
            raise TypeError('Content cannot be none')
        with self._operate_with_new_key(key) as s3_key:
            params = self._prepare_params_for_set(policy, content_type, meta_data)
            s3_key.put(Body=content, **params)
            self.logger.debug(f'KeyUploaded: {key}')

    def copy_folder(self, src_path, dst_path):
        with self._operate_with_bucket() as bucket:
            keys = list(bucket.objects.filter(Prefix=src_path))
            # TODO: Remove copied keys in case one fails
            for src_key in keys:
                dst_key = dst_path + "/" + src_key.name.split("/")[-1]
                bucket.copy_key(dst_key, self.bucket_name, src_key.name, preserve_acl=True)
                self.logger.debug(f'KeyCopied: {src_key} to {dst_key}')

    def copy_key(self, src_key_name, dst_key_name, policy=None, content_type=None, meta_data=None):
        # Ref: https://boto3.readthedocs.io/en/latest/guide/s3.html#copies
        with self._operate_with_new_key(dst_key_name) as s3_key:
            # HACK: The leading slash was added for the tests to work.
            #       We have done the operation using copy_from instead of copy as it supports
            #       the passing of the CopySource as an string. Passing it as a dictionary didn't work
            #       with s3rver.
            #       We had a similar problem using boto2 and we are applying the same hack here.
            #
            #       Even though the issue has been fixed in the s3rver source (see https://github.com/jamhall/s3rver/pull/31)  # noqa
            #       the docker image installs version 0.0.12 (see https://hub.docker.com/r/jbergknoff/s3rver/~/dockerfile/)  # noqa
            #      which does not include the fix.
            params = self._prepare_params_for_set(policy, content_type, meta_data)
            s3_key.copy_from(CopySource='/' + self.bucket_name + '/' + src_key_name, **params)
            self.logger.debug(f'KeyCopied: {src_key_name} to {dst_key_name}')

    def move_key(self, src_key_name, dst_key_name, policy=None, content_type=None, meta_data=None):
        self.copy_key(src_key_name, dst_key_name, policy=policy, content_type=content_type, meta_data=meta_data)
        self.delete(src_key_name)
        self.logger.debug(f'KeyMoved: {src_key_name} to {dst_key_name}')

    def _prepare_params_for_set(self, policy, content_type, meta_data):
        params = {}
        if policy is not None:
            params['ACL'] = policy
        if content_type is not None:
            params['ContentType'] = content_type
        if meta_data is not None:
            params['Metadata'] = meta_data
        return params

    def get_content_type(self, key):
        with self._operate_with_key(key) as s3_key:
            return s3_key.content_type

    def get_meta_data(self, key):
        with self._operate_with_key(key) as s3_key:
            return s3_key.metadata

    def get_to_file(self, key, data_path):
        with self._operate_with_key(key, validate=False) as s3_key:
            s3_key.download_file(data_path)
            self.logger.debug(f'KeyDownloaded: {key}')

    def get(self, key):
        with self._operate_with_key(key, validate=False) as s3_key:
            data = s3_key.get()['Body'].read()
            self.logger.debug(f'KeyDownloaded: {key}')
            return data

    def generate_url_from_key(self, key_name):
        url = self.s3_conn.meta.client.generate_presigned_url(
            ClientMethod='get_object',
            Params={
                'Bucket': self.bucket_name,
                'Key': key_name
            })
        url = self._remove_parameters_from_url(url)
        self.logger.debug(f'KeyURLGenerated: url={url}, key={key_name}')
        return url

    def delete(self, key):
        with self._operate_with_key(key) as s3_key:
            s3_key.delete()
            self.logger.debug(f'KeyDeleted: {key}')

    def delete_bucket(self):
        with self._operate_with_bucket() as bucket:
            bucket.delete()

    def delete_all_objects(self):
        with self._operate_with_bucket() as bucket:
            bucket.objects.all().delete()

    def delete_non_empty_bucket(self):
        self.delete_all_objects()
        self.delete_bucket()

    def delete_folder(self, key):
        with self._operate_with_bucket() as bucket:
            objects = list(bucket.objects.filter(Prefix=key))
            keys = [{'Key': object.key} for object in objects]
            bucket.delete_objects(Delete={
                'Objects': keys
            })
            self.logger.debug(f'KeyDeleted: {key}')

    def list(self, key=None):
        # i. Iterates through the keys that match the given prefix `key`
        # ii. Removes the prefix and returns the text until the first slash
        # (so only items of the _first level_ are returned).
        # iii. Ignores empty names (which happens if `bucket.list` returns
        # `normalized_key`).
        # WARNING: this method returns a list with all the elements, which
        # might be very inefficient for big resultsets. To iterate efficiently
        # through the files, please use `tree`.
        with self._operate_with_bucket() as bucket:
            normalized_key = normalize_path(key)
            result = set()
            for key in bucket.objects.filter(Prefix=normalized_key):
                keyname = key.key[len(normalized_key):].split('/')[0]
                if keyname:
                    result.add(keyname)
            return list(result)

    def tree(self, key=None):
        # i. Iterates through the keys that match the given prefix `key`
        # ii. Removes the prefix and returns the remaining text (so the
        # relative route from `key` is returned).
        # iii. Ignores empty names (which happens if `bucket.list` returns
        # `normalized_key`).
        with self._operate_with_bucket() as bucket:
            normalized_key = normalize_path(key)
            for key in bucket.objects.filter(Prefix=normalized_key):
                keyname = key.key[len(normalized_key):]
                if keyname:
                    yield keyname

    def bucket_exists(self):
        # try:
        #     self.s3_conn.head_bucket(Bucket=self.bucket_name)
        #     return True
        # except botocore.exceptions.ClientError as e:
        #     error_code = int(e.response['Error']['Code'])
        #     if error_code == 403:
        #         return True
        #     elif error_code == 404:
        #         return False
        #     else:
        #         raise
        bucket = self.s3_conn.Bucket(self.bucket_name)
        return True if bucket.creation_date else False

    def create_bucket(self):
        try:
            self.s3_conn.create_bucket(
                Bucket=self.bucket_name,
                CreateBucketConfiguration={
                    'LocationConstraint': self.region,
                })
        except botocore.exceptions.ClientError as e:
            if not check_client_error_code(e, 'BucketAlreadyExists'):
                raise BucketCreationError("We cannot create bucket. {bucket_name}. {exception}".format(
                    bucket_name=self.bucket_name,
                    exception=str(e)))

    def exist_key(self, key):
        try:
            self.s3_conn.Object(self.bucket_name, key).load()
            return True
        except botocore.exceptions.ClientError as e:
            if check_client_error_code(e, '404'):
                return False
            raise

    @contextmanager
    def _operate_with_bucket(self):
        yield self.s3_conn.Bucket(self.bucket_name)

    @contextmanager
    def _operate_with_key(self, key, validate=True):
        """ It returns a key (by default validated)
        The validation is achieved retrieving the metadata as it is what boto2 was doing and it is what our code
        base expects (The alternative would be using get()).
        Have in mind that the S3 metadata propagation is asyncronous. That is the reason why in our implementation
        of most calls we are not validating (like in the get()).
        """
        try:
            key = self._normalize_key(key)
            key = self._sanitize_key(key)
            bucket = self.s3_conn.Bucket(self.bucket_name)
            key = bucket.Object(key)
            if validate:
                key.load()
            yield key
        except botocore.errorfactory.ClientError as e:
            if check_client_error_code(e, 'NoSuchKey') or check_client_error_code(e, '404'):
                raise KeyNotFoundError("S3 key not found b: {} k: {}".format(self.bucket_name, key))
            raise

    @contextmanager
    def _operate_with_new_key(self, key):
        key = self._normalize_key(key)
        key = self._sanitize_key(key)
        bucket = self.s3_conn.Bucket(self.bucket_name)
        yield bucket.Object(key)

    def _remove_parameters_from_url(self, url):
        parsed_url = urlparse(url)
        return parsed_url.scheme + '://' + parsed_url.netloc + parsed_url.path

    def _sanitize_key(self, key):
        return key.encode('ascii', 'replace').decode('ascii')

    def _normalize_key(self, key):
        # boto2 had a different behaviour when there was a leading slash
        # In order to maintain compatibility with previous code we need to remove it
        # when using boto3
        return key[1:] if key[0] == '/' else key


def _save_str_to_file_in_s3(object_repository, filename, data):
    _, data_path_temp = tempfile.mkstemp()
    with open(data_path_temp, 'w') as f:
        f.write(data)
    object_repository.set_from_file(filename, data_path_temp)


def normalize_path(path: str) -> str:
    if path is None:
        return ''
    elif path is '' or path.endswith('/'):
        return path
    else:
        return path + '/'


def check_client_error_code(client_error, code):
    return client_error.response['Error']['Code'] == code