import os
import sys
import time
import datetime
import numpy as np
import pandas as pd
import math
import scipy.stats as stats
import statsmodels.regression.linear_model as sm
import warnings

from sklearn import preprocessing
from sklearn.ensemble import RandomForestRegressor
from sklearn.model_selection import train_test_split, KFold
from sklearn.linear_model import Lasso, LassoCV, LinearRegression
from sklearn.metrics import mean_squared_error
from minepy import MINE
from dateutil.relativedelta import *
from joblib import Parallel, delayed

from pyhive import presto
from pyhive import hive
from pyspark.sql import SparkSession
from pyspark.sql import Row
from pyspark.sql.types import *
# Removed: from hdfs import InsecureClient (now using MinIO/S3)
from concurrent.futures import ProcessPoolExecutor
import multiprocessing

# Dask for distributed Random Forest
from dask.distributed import Client
# from dask_kubernetes import KubeCluster, make_pod_spec  # Not needed for local execution
from joblib import parallel_backend

with warnings.catch_warnings():
    warnings.filterwarnings("ignore", category=DeprecationWarning)

ERROR_CODE_01 = '-1'
ERROR_CODE_02 = '-2'
ERROR_CODE_03 = '-3'
ERROR_CODE_04 = '1'
ERROR_CODE_05 = '-4'
SUCCESS_CODE_01 = '0'
SUCCESS_CODE_02 = '2'

def create_insert_value(*str_values):
    # Get current date for dt partition (format: YYYYMMDD)
    dt_partition = datetime.datetime.now().strftime("%Y%m%d")
    str_query = " ("
    for idx, value in enumerate(str_values):
        str_query += "'" + str(value) + "',"
    # Add dt partition value at the end
    str_query += "'" + dt_partition + "')"
    return str_query

def cal_adjusted_r_squared(r_squared, n, p):
    value = 1 - (((1 - r_squared) * (n - 1)) / (n - p - 1))
   #value = 1 - (((1 - r_squared) * (n - 1) / (n - p - 1)))
    return value

def bootstrap_rf(X, Y, test_size=0.3, n_est=100):
    X_train, X_test, y_train, y_test = train_test_split(X, Y, 
                                                        test_size=test_size,
                                                        random_state=np.random.randint(0, 999999))
    start_time = (datetime.datetime.now())
    rf_regressor = RandomForestRegressor(n_estimators=n_est,
                                         oob_score=True,
                                         random_state=np.random.randint(0, 999999),
                                         n_jobs=-1,
                                         warm_start=True).fit(X_train, y_train.values.ravel())
    end_time = (datetime.datetime.now())

    feature_impt = rf_regressor.feature_importances_
    feature_impt_std = np.std([tree.feature_importances_ for tree in rf_regressor.estimators_], axis=0)

    rfc_result = pd.DataFrame().assign(**{'variable': X_train.columns,
                                          'importance': feature_impt,
                                          'std': feature_impt_std})

    scaled_importance = rfc_result['importance'].astype(np.float64) / rfc_result['std'].astype(np.float64)
    scaled_importance = scaled_importance.astype(np.float64)
 
    rfc_result = rfc_result.assign(**{'s': scaled_importance,
                                            's_rank': scaled_importance.rank(),
                                            'i_rank': rfc_result.importance.rank()})
    return rfc_result

def lasso_scores(alpha, train_x, train_y, test_x, test_y):
    lasso = Lasso(alpha=alpha,
                  tol=0.0005,
                  max_iter=1000,
                  random_state=42).fit(train_x, train_y)
    train_score = lasso.score(train_x, train_y)
    test_score = lasso.score(test_x, test_y)
    return alpha, train_score, test_score

def create_lasso_dataset(df, features):
    data = df[features]
    data = data.loc[:, data.std() > .0]
    data = pd.DataFrame(preprocessing.normalize(data, axis=0), columns=data.columns)
    return data

def evaluate_feature_addition(column, included, X, y):
    """병렬 평가용 헬퍼 함수"""
    train_features = included + [column]
    train_x = X[train_features]
    model = LinearRegression(fit_intercept=False, n_jobs=-1)
    model.fit(train_x, y)
    score = model.score(train_x, y)

    if (len(X) - len(model.coef_) - 1) == 0:
        adj_score = score
    else:
        adj_score = cal_adjusted_r_squared(score, len(X), len(model.coef_))
    return column, adj_score

def step_forward_k_fold(X, y, k=5, tol=0.000, verbose=True):
    """병렬화된 Step Forward K-Fold Selection"""
    kf = KFold(n_splits=k)
    kf.get_n_splits(X)
    included = list()
    total_result = pd.DataFrame(
        columns=["# of var", "model", "var_included", "coefs", "Adj_Rsquared", "K-Fold Adj_Rsquared"])

    curr_best_score = -np.inf
    curr_best_kfold_score = -np.inf
    j = 0

    while True:
        changed = False
        included.sort()
        excluded = [feature for feature in list(X.columns) if feature not in included]
        excluded.sort()

        if len(excluded) == 0:
            print("Stopping the Forward Selection")
            break

        # === 병렬 평가: 모든 excluded feature를 동시에 평가 ===
        if verbose:
            print(f"[Step {len(included)+1}] Evaluating {len(excluded)} features in parallel...")
        results = Parallel(n_jobs=-1)(
            delayed(evaluate_feature_addition)(col, included, X, y) for col in excluded
        )
        new_score_dict = {col: score for col, score in results}
        best_feature, best_adj_score = max(results, key=lambda x: x[1])

        if verbose:
            print(f"  Scores: {new_score_dict}")
            print(f"  Best feature: {best_feature} (Adj-R²={best_adj_score:.6f})")

        if (best_adj_score - curr_best_score) > tol:
            included.append(best_feature)
            curr_best_score = best_adj_score
            changed = True

        cv_result = pd.DataFrame(columns=['k_fold', 'Adj_Rsquared'])
        i = 0

        if len(X) >= 30:
            for _, test_idx in kf.split(X, y):
                X_test = X.iloc[test_idx,]
                y_test = y.iloc[test_idx,]
                model = LinearRegression(fit_intercept=False,
                                         n_jobs=-1)
                model.fit(X_test[included + [best_feature]], y_test)
                score = model.score(X_test[included + [best_feature]], y_test)
                cv_score = cal_adjusted_r_squared(score, len(X_test), len(model.coef_))
                cv_inner_res = pd.DataFrame({'k_fold': [i],
                                              'Adj_Rsquared': [cv_score]})
                cv_result = pd.concat([cv_result, cv_inner_res])
                i += 1
        else:
            model = LinearRegression(fit_intercept=False,
                                     n_jobs=-1)
            model.fit(X[included + [best_feature]], y)
            score = model.score(X[included + [best_feature]], y)
            if (len(X) - len(model.coef_) - 1) == 0:
                cv_score = score
            else:
                cv_score = cal_adjusted_r_squared(score, len(X), len(model.coef_))
            cv_inner_res = pd.DataFrame({"k_fold": [0], "Adj_Rsquared": [cv_score]})
            cv_result = pd.concat([cv_result, cv_inner_res])

        kf_score = cv_result['Adj_Rsquared'].mean()
        curr_model = LinearRegression(fit_intercept=False,
                                      n_jobs=-1).fit(X[included], y)                            

        if kf_score > curr_best_kfold_score:
            curr_best_kfold_score = kf_score
        else:
            j += 1
        if changed and verbose:
            pass
        if j == 10:
            changed = False
        if not changed:
            break

        result = pd.DataFrame(
            {'# of var': len(included),
              'model': curr_model,
              'var_included': [included.copy()],
              'coefs': [curr_model.coef_.copy()],
              'Adj_Rsquared': best_adj_score,
              'K-Fold Adj_Rsquared': kf_score})
        total_result = pd.concat([total_result, result])
    total_result.reset_index(inplace=True)
    return total_result
def test_oracle_connection(spark):
    """Test Oracle JDBC connection (간단한 현재 시간 조회)"""
    try:
        # Check if Oracle JAR exists
        ORACLE_JDBC_JAR = os.getenv("ORACLE_JDBC_JAR", "/root/oracle_jdbc/jdbc/ojdbc8.jar")
        if not os.path.exists(ORACLE_JDBC_JAR):
            print(f"[SKIP] Oracle JDBC JAR not found: {ORACLE_JDBC_JAR}")
            return False

        ORACLE_JDBC_URL = os.getenv("ORACLE_JDBC_URL", "jdbc:oracle:thin:@//localhost:1521/FREEPDB1")
        ORACLE_USER = os.getenv("ORACLE_USER", "system")
        ORACLE_PASSWORD = os.getenv("ORACLE_PASSWORD", "Oracle123")
        ORACLE_DRIVER = "oracle.jdbc.OracleDriver"

        print("[INFO] Testing Oracle connection...")

        # Query: Get current date/time from Oracle
        df_oracle = spark.read \
            .format("jdbc") \
            .option("url", ORACLE_JDBC_URL) \
            .option("dbtable", "(SELECT SYSDATE AS current_date, CURRENT_TIMESTAMP AS current_timestamp FROM dual)") \
            .option("user", ORACLE_USER) \
            .option("password", ORACLE_PASSWORD) \
            .option("driver", ORACLE_DRIVER) \
            .load()

        print("[SUCCESS] Oracle connection test passed!")
        print("Oracle Date/Time:")
        df_oracle.show(truncate=False)

        return True

    except Exception as e:
        print(f"[ERROR] Oracle connection failed: {e}")
        return False


def update_oracle_status(spark, job_id, area, status, current_step, message, progress):
    """Update regression status in Oracle using python-oracledb"""

    import oracledb
    import os

    # Oracle 연결 정보
    oracle_user = os.getenv("ORACLE_USER", "system")
    oracle_password = os.getenv("ORACLE_PASSWORD", "Oracle123")
    oracle_dsn = os.getenv("ORACLE_DSN", "localhost:1521/FREEPDB1")

    try:
        # 연결 생성
        connection = oracledb.connect(
            user=oracle_user,
            password=oracle_password,
            dsn=oracle_dsn
        )
        cursor = connection.cursor()

        # MERGE SQL (UPSERT) - jobid가 키
        merge_sql = """MERGE INTO regression_status t
USING (SELECT :1 jobid FROM dual) s
ON (t.jobid = s.jobid)
WHEN MATCHED THEN
    UPDATE SET status=:2, progress=:3, update_time=SYSTIMESTAMP
WHEN NOT MATCHED THEN
    INSERT (jobid, area, status, progress, start_time)
    VALUES (:4, :5, :6, :7, SYSTIMESTAMP)"""

        # 파라미터 바인딩 (SQL Injection 방지)
        cursor.execute(merge_sql, [
            job_id,           # :1 (MATCHED 조건)
            status,           # :2 (UPDATE status)
            progress,         # :3 (UPDATE progress)
            job_id,           # :4 (INSERT jobid)
            area,             # :5 (INSERT area)
            status,           # :6 (INSERT status)
            progress          # :7 (INSERT progress)
        ])

        connection.commit()
        print(f"[ORACLE] {status} - {progress}%")

    except Exception as e:
        print(f"[ERROR] Oracle update failed: {e}")
    finally:
        if 'cursor' in locals():
            cursor.close()
        if 'connection' in locals():
            connection.close()


def select_oracle_status():
    """Select all regression status from Oracle using python-oracledb"""

    import oracledb
    import os

    # Oracle 연결 정보
    oracle_user = os.getenv("ORACLE_USER", "system")
    oracle_password = os.getenv("ORACLE_PASSWORD", "Oracle123")
    oracle_dsn = os.getenv("ORACLE_DSN", "localhost:1521/FREEPDB1")

    try:
        # 연결 생성
        connection = oracledb.connect(
            user=oracle_user,
            password=oracle_password,
            dsn=oracle_dsn
        )
        cursor = connection.cursor()

        # SELECT SQL - 전체 조회
        select_sql = """SELECT jobid, area, status, progress, start_time, update_time
                        FROM regression_status
                        ORDER BY update_time DESC"""

        # 실행 (파라미터 없음)
        cursor.execute(select_sql)

        # 결과 조회
        rows = cursor.fetchall()

        if rows:
            print(f"[ORACLE SELECT] Found {len(rows)} record(s)")
            for row in rows:
                print(f"  JobID: {row[0]}, Area: {row[1]}, Status: {row[2]}, Progress: {row[3]}%, Start: {row[4]}, Update: {row[5]}")
            return rows
        else:
            print(f"[ORACLE SELECT] No records found")
            return None

    except Exception as e:
        print(f"[ERROR] Oracle select failed: {e}")
        return None
    finally:
        if 'cursor' in locals():
            cursor.close()
        if 'connection' in locals():
            connection.close()


if __name__ == "__main__":
    total_time = time.time()

    # MinIO Configuration - use environment variable for Docker/Kubernetes
    MINIO_ENDPOINT = os.getenv("MINIO_ENDPOINT", "http://localhost:9011")
    MINIO_ACCESS_KEY = os.getenv("MINIO_ACCESS_KEY", "minioadmin")
    MINIO_SECRET_KEY = os.getenv("MINIO_SECRET_KEY", "minioadmin")

    # Oracle JDBC Configuration
    ORACLE_JDBC_URL = os.getenv("ORACLE_JDBC_URL", "jdbc:oracle:thin:@//localhost:1521/FREEPDB1")
    ORACLE_USER = os.getenv("ORACLE_USER", "system")
    ORACLE_PASSWORD = os.getenv("ORACLE_PASSWORD", "Oracle123")
    ORACLE_JDBC_JAR = os.getenv("ORACLE_JDBC_JAR", "/root/oracle_jdbc/jdbc/ojdbc8.jar")

    # Check if Oracle JAR exists
    oracle_jar_exists = os.path.exists(ORACLE_JDBC_JAR)

    # Configure Spark with Iceberg and MinIO
    spark_builder = (SparkSession
            .builder
            .appName("RegressionAnalysis")
            .config("spark.sql.extensions", "org.apache.iceberg.spark.extensions.IcebergSparkSessionExtensions")
            .config("spark.sql.catalog.iceberg", "org.apache.iceberg.spark.SparkCatalog")
            .config("spark.sql.catalog.iceberg.type", "hadoop")
            .config("spark.sql.catalog.iceberg.warehouse", "s3a://ic-ias/structured")
            .config("spark.sql.catalog.iceberg.io-impl", "org.apache.iceberg.hadoop.HadoopFileIO")
            # S3A / MinIO Configuration
            .config("spark.hadoop.fs.s3a.endpoint", MINIO_ENDPOINT)
            .config("spark.hadoop.fs.s3a.access.key", MINIO_ACCESS_KEY)
            .config("spark.hadoop.fs.s3a.secret.key", MINIO_SECRET_KEY)
            .config("spark.hadoop.fs.s3a.path.style.access", "true")
            .config("spark.hadoop.fs.s3a.connection.ssl.enabled", "false")
            .config("spark.hadoop.fs.s3a.impl", "org.apache.hadoop.fs.s3a.S3AFileSystem")
            .config("spark.hadoop.fs.s3a.aws.credentials.provider", "org.apache.hadoop.fs.s3a.SimpleAWSCredentialsProvider"))

    # Add Oracle JDBC JAR if exists
    if oracle_jar_exists:
        spark_builder = spark_builder.config("spark.jars", ORACLE_JDBC_JAR)
        print(f"[INFO] Oracle JDBC JAR loaded: {ORACLE_JDBC_JAR}")

    spark = spark_builder.getOrCreate()

    job_id = ""
    yparam = ""
    cluster_area = ""

    for i in range(1, len(sys.argv)):
        name = sys.argv[i].split(":")[0]
        value = sys.argv[i].split(":")[1]
        if (name == "jobid"): 
            job_id = value
        elif (name == "area"):
            cluster_area = value
        elif (name == "yparam"): 
            yparam = value

    if (cluster_area == "local"):
        hive_server = "localhost"
        presto_discovery = "localhost"
        minio_endpoint = MINIO_ENDPOINT  # Use environment variable
    elif (cluster_area == "ich"):
        hive_server = "ichbig-01-002"
        presto_discovery = "10.38.12.216"
        minio_endpoint = "http://icbig-00-11:9011"
    elif (cluster_area == "wxh"):
        hive_server = "wuxbigm-001-02"
        presto_discovery = "wuxbigm-004-01"
        minio_endpoint = "http://wuxbigm-001-01:9011"

    # Read from MinIO/S3 using Spark (ich, wxh only)
    input_path = f"s3a://ic-ias/structured/{job_id}"
    try:
        Total = spark.read.csv(input_path, header=True, inferSchema=True).toPandas()
        if cluster_area == "ich":
            print(f"[INFO] Successfully read from MinIO (ich): {input_path}")
        else:  # wxh
            print(f"[INFO] Successfully read from MinIO (wxh): {input_path}")

        # Update Oracle status: STARTED
        if os.getenv("UPDATE_ORACLE_STATUS", "false").lower() == "true":
            update_oracle_status(spark, job_id, cluster_area, "STARTED",
                                "Data Loading",
                                f"Data loaded: {len(Total)} rows × {len(Total.columns)} features", 0)

    except Exception as e:
        print(f"[ERROR] Failed to read from MinIO: {e}")
        sys.exit(1)

    if Total.empty:
        print("Empty")
        insertQuery = "Insert into iceberg.ias.fdc_rfimportance VALUES"
        insertQuery += create_insert_value(ERROR_CODE_01, 'error', 'error', 'error', job_id)
        spark.sql(insertQuery)
        sys.exit(0)

    elif (len(Total) < 10):
        insertQuery = "Insert into iceberg.ias.fdc_rfimportance VALUES"
        insertQuery += create_insert_value(ERROR_CODE_05, 'error', 'error', 'error', job_id)
        spark.sql(insertQuery)
        sys.exit(0)
    else:
        path = "/User/james/PycharmProjets/test1/result/"

        response = yparam
        not_used = {'END_TIME', 'END_TM', 'FAB', 'LOT_CD', 'ALIAS_LOT_ID', 'LOT_ID', 'WF_ID', 'group_key', 'module', 'recipe_id', 'oper', 'FAB(FDC)', 'LOT_CD(FDC)'}
        list_end_time = ['END_TIME']
        str_end_time = "END_TIME"

        if Total[response].std() == 0:
            insertQuery = "Insert into iceberg.ias.fdc_rfimportance VALUES"
            insertQuery += create_insert_value(ERROR_CODE_02, 'error', 'error', 'error', job_id)
            spark.sql(insertQuery)
            sys.exit(0)

        R2SM_si = Total.dropna(subset=[response])
        R2SM_si.fillna(R2SM_si.mean(numeric_only=True), inplace=True)

        x_features = [x_feature for x_feature in list(R2SM_si.columns) if x_feature not in response]
        describes = R2SM_si[x_features].describe()
        no_volatility_features = [k for k, v in describes.loc['std'].to_dict().items() if v == 0.0 or np.isnan(v)]
        x_features = [x_feature for x_feature in x_features if x_feature not in no_volatility_features]

        temp_not_used_columns = not_used.union(list_end_time)
        x_features = [x_feature for x_feature in x_features if x_feature not in temp_not_used_columns]

        X_R2SM_si = R2SM_si[x_features]
        X_R2SM_si = X_R2SM_si.dropna(axis=1)
        y_R2SM_si = R2SM_si[response]

        if X_R2SM_si.empty :
            insertQuery = "Insert into iceberg.ias.fdc_rfimportance VALUES"
            insertQuery += create_insert_value(ERROR_CODE_03, 'error', 'error', 'error', job_id)
            spark.sql(insertQuery)
            sys.exit(0)

        scaler = preprocessing.StandardScaler()
        X_R2SM_si = pd.DataFrame(scaler.fit_transform(X_R2SM_si), columns=X_R2SM_si.columns)
        y_R2SM_si = pd.DataFrame(scaler.fit_transform(y_R2SM_si.values.reshape(-1,1)), columns=[response])

        print(f"[INFO] Data loaded: {len(X_R2SM_si)} rows × {len(X_R2SM_si.columns)} features")

        if np.all(X_R2SM_si.std() == 0):
            insertQuery = "Insert into iceberg.ias.fdc_rfimportance VALUES"
            insertQuery += create_insert_value(ERROR_CODE_03, 'error', 'error', 'error', job_id)
            spark.sql(insertQuery)
            sys.exit(0)

        print(f"[INFO] Step 1: Random Forest Bootstrap (10 iterations) with Dask...")

        # === Pre-screening 파라미터 설정 (README 1829-1951행 참고) ===
        # 데이터가 1만 건이면 feature 수가 많을 수 있으므로 상위 N개로 제한
        RF_TOP_N = 100  # README에서 제안된 대규모 데이터용 설정
        LASSO_TOP_N = 50  # Lasso에서 선택할 최대 feature 수

        # === Dask 분산 처리 적용 ===
        # Kubernetes 환경 확인
        USE_KUBERNETES = os.getenv("USE_KUBERNETES", "false").lower() == "true"
        DASK_WORKER_IMAGE = os.getenv("DASK_WORKER_IMAGE", "pyspark-miniconda:latest")
        DASK_NAMESPACE = os.getenv("DASK_NAMESPACE", "default")
        DASK_WORKER_N_WORKERS = int(os.getenv("DASK_WORKER_N_WORKERS", "2"))
        DASK_WORKER_THREADS = int(os.getenv("DASK_WORKER_THREADS", "2"))
        DASK_WORKER_MEMORY = os.getenv("DASK_WORKER_MEMORY", "2GB")

        if USE_KUBERNETES:
            # Kubernetes 분산 클러스터 환경: Worker Pod 자동 생성
            print(f"[INFO] Using Dask Kubernetes (workers: {DASK_WORKER_N_WORKERS}, image: {DASK_WORKER_IMAGE})")

            # Worker Pod 사양 정의 (2022.10.1 API)
            worker_pod_template = make_pod_spec(
                image=DASK_WORKER_IMAGE,
                memory_limit=DASK_WORKER_MEMORY,
                env={
                    'PYTHONPATH': '/opt/conda/envs/hynix/lib/python3.9/site-packages',
                    'CONDA_DEFAULT_ENV': 'hynix',
                    'PATH': '/opt/conda/envs/hynix/bin:$PATH',
                }
            )

            # KubeCluster 생성 (2022.10.1 API)
            cluster = KubeCluster(
                pod_template=worker_pod_template,
                n_workers=DASK_WORKER_N_WORKERS,
                name="dask-rf-worker",
                namespace=DASK_NAMESPACE,
                silence_logs=True,
                shutdown_on_close=True
            )

            # Worker가 시작될 때까지 더 오래 기다림
            client = Client(cluster, timeout=60)
            try:
                # Worker가 연결될 때까지 기다림
                client.wait_for_workers(DASK_WORKER_N_WORKERS, timeout=60)
                with parallel_backend('dask'):
                    out = Parallel(n_jobs=-1)(delayed(bootstrap_rf)(X_R2SM_si, y_R2SM_si, 0.3) for _ in range(10))
            finally:
                client.close()
                cluster.close()
        else:
            # 로컬 개발 환경: 기존 방식 그대로 사용
            print(f"[INFO] Using Dask Local (workers: {DASK_WORKER_N_WORKERS})")
            client = Client(n_workers=DASK_WORKER_N_WORKERS,
                          threads_per_worker=DASK_WORKER_THREADS,
                          memory_limit=DASK_WORKER_MEMORY,
                          silence_logs=True)
            try:
                with parallel_backend('dask'):
                    out = Parallel(n_jobs=-1)(delayed(bootstrap_rf)(X_R2SM_si, y_R2SM_si, 0.3) for _ in range(10))
            finally:
                client.close()

        result_rfc = pd.concat(out, axis=0, ignore_index=True)

        stats_of_importance_each_variable = result_rfc.groupby('variable')['importance'].describe()
        stats_of_importance_each_variable['standard_importance_rank'] = stats_of_importance_each_variable['mean'].rank(ascending=False)
        stats_of_importance_each_variable.sort_values(by='standard_importance_rank',
                                                      ascending=True,
                                                      inplace=True)
        response_df = R2SM_si[[response, str_end_time]].reset_index()

        response_df.loc[:, 'TIME'] = pd.to_datetime(response_df.loc[:, str_end_time])
        response_df.loc[:, 'TIME1'] = pd.to_numeric(response_df.loc[:, 'TIME'])

        final_raw_df = pd.concat([X_R2SM_si, response_df], axis=1)
        rf_feature_impt_mean_over_zero = stats_of_importance_each_variable.query('mean > 0')
        print(f"[INFO] Random Forest completed: {len(rf_feature_impt_mean_over_zero)} features selected")

        # Update Oracle status: RF completed
        if os.getenv("UPDATE_ORACLE_STATUS", "false").lower() == "true":
            update_oracle_status(spark, job_id, cluster_area, "RF_COMPLETED",
                                "Random Forest Bootstrap",
                                f"Selected {len(rf_feature_impt_mean_over_zero)} features", 30)

        # === Pre-screening: 상위 N개 feature만 선택 ===
        if len(rf_feature_impt_mean_over_zero) > RF_TOP_N:
            rf_feature_impt_mean_over_zero = rf_feature_impt_mean_over_zero.nlargest(RF_TOP_N, 'mean')
            print(f"[INFO] Pre-screening: Selected top {RF_TOP_N} features from Random Forest")
        else:
            print(f"[INFO] All {len(rf_feature_impt_mean_over_zero)} RF features retained (less than RF_TOP_N)")

        features = list(rf_feature_impt_mean_over_zero.index.values)
        XX_R2SM_si = create_lasso_dataset(X_R2SM_si, features)
        candidate_alphas = np.logspace(-10, -3, 100)  # 0.0000000001 ~ 0.001

        print(f"[INFO] Step 2: LASSO Regression started with {len(XX_R2SM_si.columns)} features...")

        X_train, X_test, y_train, y_test = train_test_split(XX_R2SM_si, y_R2SM_si, test_size=0.3, random_state=42)
        lasso_performance = \
            Parallel(n_jobs=-1, prefer='threads')(
                delayed(lasso_scores)(alpha, X_train, y_train, X_test, y_test) for alpha in candidate_alphas)
        lasso_performance = pd.DataFrame(lasso_performance, columns=['alpha', 'train_score', 'test_score'])
        optimal_alpha = lasso_performance.iloc[lasso_performance.iloc[:, 2].idxmax(), 0]

        print("Optimal alpha for LASSO: {: .6f}".format(optimal_alpha))

        optimal_lasso_model = Lasso(fit_intercept=False,
                                    tol=0.0005,
                                    alpha=optimal_alpha).fit(XX_R2SM_si, y_R2SM_si)
                
        score = optimal_lasso_model.score(XX_R2SM_si, y_R2SM_si)
        adj_score = cal_adjusted_r_squared(score,
                                        len(XX_R2SM_si),
                                        len(optimal_lasso_model.coef_[abs(optimal_lasso_model.coef_) > .0]))

        lasso_coefficients = optimal_lasso_model.coef_.tolist()
        selected_coefficients_idx = np.where(np.abs(lasso_coefficients) != 0)[0].tolist()
        selected_coefficients_values = np.take(lasso_coefficients, selected_coefficients_idx)

        # === Pre-screening: 상위 N개 coefficient만 선택 ===
        if len(selected_coefficients_idx) > LASSO_TOP_N:
            # 절대값이 큰 순서대로 상위 N개 선택
            top_idx = np.argsort(np.abs(selected_coefficients_values))[-LASSO_TOP_N:]
            selected_coefficients_idx = np.array(selected_coefficients_idx)[top_idx].tolist()
            selected_coefficients_values = np.take(lasso_coefficients, selected_coefficients_idx)
            print(f"[INFO] Pre-screening: Selected top {LASSO_TOP_N} Lasso features")
        else:
            print(f"[INFO] All {len(selected_coefficients_idx)} Lasso features retained")

        print(f"[INFO] LASSO completed: {len(selected_coefficients_idx)} features with non-zero coefficients")

        # Update Oracle status: LASSO completed
        if os.getenv("UPDATE_ORACLE_STATUS", "false").lower() == "true":
            update_oracle_status(spark, job_id, cluster_area, "LASSO_COMPLETED",
                                "LASSO Regression",
                                f"Selected {len(selected_coefficients_idx)} features", 50)

        if len(selected_coefficients_idx) == 0:
            insertQuery = "Insert into iceberg.ias.fdc_rfimportance VALUES"
            for idx, row in stats_of_importance_each_variable.iterrows():
                step = idx.split(".")[0]
                param = idx.split(".")[1]
                insertQuery += create_insert_value(ERROR_CODE_04,
                                                   param, step, row['mean'], job_id) + ","
            insertQuery = insertQuery[0:len(insertQuery)-1]
            spark.sql(insertQuery)
        else:
            lasso_features_1st = pd.DataFrame().assign(**{
                'variable': XX_R2SM_si.columns[selected_coefficients_idx],
                'score': selected_coefficients_values,
                'abs_score': abs(selected_coefficients_values)
            })
            lasso_features_1st.sort_values(by='abs_score', ascending=False, inplace=True)

            str_time_0 = time.strftime("%Y%m%d-%H%M%S")

            features = list(lasso_features_1st['variable'])
            XX_R2SM_si = create_lasso_dataset(X_R2SM_si, features)

            start_time_5fold = datetime.datetime.now()
            print(f"[INFO] Step 3: Step Forward K-Fold Selection started with {len(XX_R2SM_si.columns)} features...")
            total_result = step_forward_k_fold(XX_R2SM_si, y_R2SM_si)
            end_time_5fold = datetime.datetime.now()
            elapsed_5fold = (end_time_5fold - start_time_5fold).total_seconds()

            max_adj_rsquared_idx = total_result['K-Fold Adj_Rsquared'].idxmax()

            lasso_forward = pd.DataFrame().assign(**{
                'Variable': total_result['var_included'][max_adj_rsquared_idx],
                'Score': total_result['coefs'][max_adj_rsquared_idx].ravel(),
                'ABS_Score': np.abs(total_result['coefs'][max_adj_rsquared_idx].ravel())})
            lasso_forward.sort_values(by='ABS_Score', ascending=False, inplace=True)
            print(f"[INFO] Step Forward completed in {elapsed_5fold:.2f}s: {len(lasso_forward['Variable'].tolist())} features selected")

            # Update Oracle status: Step Forward completed
            if os.getenv("UPDATE_ORACLE_STATUS", "false").lower() == "true":
                update_oracle_status(spark, job_id, cluster_area, "STEP_FORWARD_COMPLETED",
                                    "Step Forward Selection",
                                    f"Selected {len(lasso_forward['Variable'].tolist())} features", 70)

            str_time_0 = time.strftime("%Y%m%d-%H%M%S")

            model = total_result['model'][max_adj_rsquared_idx]
            score = model.score(XX_R2SM_si[total_result['var_included'][max_adj_rsquared_idx]], y_R2SM_si)

            non_zero_coefficients = model.coef_[abs(model.coef_) > 0]
            train_adj_rsquared = cal_adjusted_r_squared(score, len(XX_R2SM_si), len(non_zero_coefficients))

            start_time_corr = datetime.datetime.now()

            lasso_forward_features = list(lasso_forward.Variable)
            rf_features = rf_feature_impt_mean_over_zero.index.values.ravel()


            def compute_spearman_correlations(feature_pair):
                lasso_feature_name, rf_feature_name = feature_pair
                corr_value, p_value = stats.spearmanr(final_raw_df[rf_feature_name],
                                                      final_raw_df[lasso_feature_name])
                res_corr_value, res_p_value = stats.spearmanr(final_raw_df[[response]],
                                                              final_raw_df[rf_feature_name])
                return (lasso_feature_name, rf_feature_name,
                        abs(corr_value), p_value, abs(res_corr_value), res_p_value)
            
            def compute_mine_correlation_parallel(feature):
                mine = MINE()
                mine.compute_score(final_raw_df[response], final_raw_df[feature])
                mic_value = mine.mic()
                return (feature, mic_value)

            feature_pairs = [(lasso_feature_name, rf_feature_name)
                             for lasso_feature_name in lasso_forward_features
                             for rf_feature_name in rf_features]

            print(f"[INFO] Step 4: MIC/Spearman Correlation started with {len(feature_pairs)} pairs (sequential processing)...")
            # Sequential processing (faster for small datasets)
            spearman_result = [compute_spearman_correlations(pair) for pair in feature_pairs]
            print(f"[INFO] Spearman correlation completed: {len(spearman_result)} pairs processed")

            rf_and_lasso_feature_corr = pd.DataFrame(spearman_result,
                                                     columns=['Final', 'Suggestion',
                                                              'Absolute Spearman Corr.(Final-Suggestion)',
                                                              'P-value of Absolute Spearman Corr.(Final-Suggestion)',
                                                              'Absolute Spearman Corr.(Target-Suggestion)',
                                                              'P-value of Absolute Spearman Corr.(Target-Suggestion)'])

            rf_and_lasso_feature_corr = rf_and_lasso_feature_corr.loc[
                                        rf_and_lasso_feature_corr['Absolute Spearman Corr.(Target-Suggestion)'] >
                                        rf_and_lasso_feature_corr['Absolute Spearman Corr.(Target-Suggestion)'].median(), :]


            unique_features = rf_and_lasso_feature_corr.iloc[:, 1].unique()
            print(f"[INFO] MIC calculation started for {len(unique_features)} unique features (sequential processing)...")
            # Sequential processing (faster for small datasets)
            mine_results = [compute_mine_correlation_parallel(feat) for feat in unique_features]
            mine_results = dict(mine_results)
            print(f"[INFO] MIC calculation completed: {len(mine_results)} features processed")

            # Update Oracle status: MIC completed
            if os.getenv("UPDATE_ORACLE_STATUS", "false").lower() == "true":
                update_oracle_status(spark, job_id, cluster_area, "MIC_COMPLETED",
                                    "MIC/Spearman Correlation",
                                    f"MIC calculation completed for {len(mine_results)} features", 90)

            mic_total_result = [mine_results[rf_and_lasso_feature_corr.iloc[i, 1]]
                                for i in range(rf_and_lasso_feature_corr.shape[0])]


            rf_and_lasso_feature_corr['Maximal Information Coeff.(Target-Suggestion)'] = mic_total_result
            mean_of_corr = (rf_and_lasso_feature_corr['Absolute Spearman Corr.(Final-Suggestion)'] +
                            rf_and_lasso_feature_corr['Absolute Spearman Corr.(Target-Suggestion)'])/2
            mean_of_corr_mic = (rf_and_lasso_feature_corr['Absolute Spearman Corr.(Final-Suggestion)'] +
                                rf_and_lasso_feature_corr['Maximal Information Coeff.(Target-Suggestion)'])/2

            rf_and_lasso_feature_corr['mean_of_corr'] = mean_of_corr
            rf_and_lasso_feature_corr['mean_of_corr(mic)'] = mean_of_corr_mic
            rf_and_lasso_feature_corr.sort_values(by='mean_of_corr(mic)', ascending=False, inplace=True)
            rf_and_lasso_feature_corr = rf_and_lasso_feature_corr.reset_index().drop(columns=['index'])

            mic_max_idx_each_feature = sorted(rf_and_lasso_feature_corr.groupby('Suggestion')['mean_of_corr(mic)'].idxmax().values)
            rf_and_lasso_feature_corr_final = rf_and_lasso_feature_corr.iloc[mic_max_idx_each_feature, :].reset_index()

            end_time_corr = datetime.datetime.now()

            result_code = SUCCESS_CODE_01 if rf_and_lasso_feature_corr_final.size > 0 else SUCCESS_CODE_02

            print(f"[INFO] Step 5: Saving results to Hive tables started...")
            print(f"[INFO] Final correlation results: {len(rf_and_lasso_feature_corr_final)} rows")

            labs = math.ceil(len(stats_of_importance_each_variable)/500.0)
            for i in range(labs):
                if (i + 1) * 500 < len(stats_of_importance_each_variable):
                    max_y = (i + 1) * 500
                else:
                    max_y = len(stats_of_importance_each_variable)
                insertQuery = "Insert into iceberg.ias.fdc_rfimportance VALUES"
                for idx, row in stats_of_importance_each_variable[i*500:max_y].iterrows():
                    step = idx.split(".")[0]
                    param = idx.split(".")[1]
                    insertQuery += create_insert_value(result_code, param, step, row['mean'], job_id) +","
            insertQuery = insertQuery[0:len(insertQuery)-1]
            spark.sql(insertQuery)

            labs = math.ceil(len(lasso_forward)/500.0)
            for i in range(labs):
                if (i + 1) * 500 < len(lasso_forward):
                    max_y = (i + 1) * 500
                else:
                    max_y = len(lasso_forward)

                insertQuery = "Insert into iceberg.ias.fdc_coeff VALUES"
                for idx, row in lasso_forward[i * 500:max_y].iterrows():
                    insertQuery += create_insert_value(result_code,
                                                       row['Variable'], str(row['Score']), str(row['ABS_Score']),
                                                       job_id) + ","

                insertQuery = insertQuery[0:len(insertQuery)-1]
                spark.sql(insertQuery)

                laps = math.ceil(len(rf_and_lasso_feature_corr_final) / 500.0)
                for i in range(laps):
                    if(i + 1) * 500 < len(rf_and_lasso_feature_corr_final):
                        max_y = (i + 1) * 500
                    else:
                        max_y = len(rf_and_lasso_feature_corr_final)

                    insertQuery = "Insert into iceberg.ias.fdc_result VALUES"
                    for idx, row in rf_and_lasso_feature_corr_final[i*500:max_y].iterrows():
                        insertQuery += create_insert_value('0',
                                                  row['Final'], row['Suggestion'],
                                                  str(row['Absolute Spearman Corr.(Final-Suggestion)']),
                                                  str(row['Absolute Spearman Corr.(Target-Suggestion)']),
                                                  str(row['mean_of_corr']),
                                                  str(row['Maximal Information Coeff.(Target-Suggestion)']),
                                                  str(row['mean_of_corr(mic)']),
                                                  job_id) + ","
                    insertQuery =  insertQuery[0:len(insertQuery)-1]
                    spark.sql(insertQuery)

            total_elapsed = time.time() - total_time
            print(f"[INFO] Regression analysis completed successfully in {total_elapsed:.2f}s")
            print(f"[INFO] Result code: {result_code}")

            # Update Oracle status: COMPLETED
            if os.getenv("UPDATE_ORACLE_STATUS", "false").lower() == "true":
                update_oracle_status(spark, job_id, cluster_area, "COMPLETED",
                                    "Regression Analysis",
                                    f"Completed in {total_elapsed:.2f}s - {len(rf_and_lasso_feature_corr_final)} features selected",
                                    100)
    
    # Optional: Test Oracle connection
    if os.getenv("TEST_ORACLE", "false").lower() == "true":
        print("[INFO] Oracle connection test enabled...")
        test_oracle_connection(spark)

    # Optional: Select Oracle status
    if os.getenv("SELECT_ORACLE_STATUS", "false").lower() == "true":
        print("[INFO] Oracle SELECT enabled...")
        select_oracle_status()




                                                     
                                                     
                                    
                


