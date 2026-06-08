#!/usr/bin/env python3
"""
PySpark Oracle JDBC Connection Example
Oracle Free 23ai (XE) 데이터베이스 연결
"""

from pyspark.sql import SparkSession
from pyspark.conf import SparkConf

# Oracle JDBC 연결 설정
# Oracle Free 23ai는 Service Name 사용 필요
ORACLE_JDBC_URL = "jdbc:oracle:thin:@//localhost:1521/FREEPDB1"
ORACLE_DRIVER = "oracle.jdbc.OracleDriver"
DB_USER = "system"
DB_PASSWORD = "Oracle123"
JDBC_JAR = "/root/oracle_jdbc/jdbc/ojdbc11.jar"


def create_spark_session():
    """Spark 세션 생성"""
    conf = SparkConf() \
        .setAppName("OracleJDBC") \
        .set("spark.jars", JDBC_JAR) \
        .set("spark.driver.extraClassPath", JDBC_JAR) \
        .set("spark.executor.extraClassPath", JDBC_JAR)

    spark = SparkSession.builder.config(conf=conf).getOrCreate()
    return spark


def test_oracle_connection(spark):
    """Oracle 연결 테스트"""
    print("=" * 60)
    print("Oracle JDBC Connection Test")
    print("=" * 60)
    print(f"JDBC URL: {ORACLE_JDBC_URL}")
    print(f"Driver: {ORACLE_DRIVER}")
    print(f"JAR: {JDBC_JAR}")
    print("-" * 60)

    # JDBC 연결 속성
    connection_props = {
        "user": DB_USER,
        "password": DB_PASSWORD,
        "driver": ORACLE_DRIVER,
        "url": ORACLE_JDBC_URL
    }

    try:
        # 현재 날짜/시간 조회 (연결 테스트)
        print("\n[Test 1] 현재 날짜/시간 조회:")
        df_sysdate = spark.read \
            .format("jdbc") \
            .option("url", ORACLE_JDBC_URL) \
            .option("dbtable", "(SELECT SYSDATE AS current_date, CURRENT_TIMESTAMP AS current_timestamp FROM dual)") \
            .option("user", DB_USER) \
            .option("password", DB_PASSWORD) \
            .option("driver", ORACLE_DRIVER) \
            .load()

        df_sysdate.show(truncate=False)

        # Oracle 버전 확인
        print("\n[Test 2] Oracle 버전 확인:")
        df_version = spark.read \
            .format("jdbc") \
            .option("url", ORACLE_JDBC_URL) \
            .option("dbtable", "(SELECT * FROM v$version WHERE rownum <= 5)") \
            .option("user", DB_USER) \
            .option("password", DB_PASSWORD) \
            .option("driver", ORACLE_DRIVER) \
            .load()

        df_version.show(truncate=False)

        # 데이터베이스 이름 확인
        print("\n[Test 3] 데이터베이스 이름 확인:")
        df_dbname = spark.read \
            .format("jdbc") \
            .option("url", ORACLE_JDBC_URL) \
            .option("dbtable", "(SELECT NAME, DATABASE_ROLE, OPEN_MODE FROM v$database)") \
            .option("user", DB_USER) \
            .option("password", DB_PASSWORD) \
            .option("driver", ORACLE_DRIVER) \
            .load()

        df_dbname.show(truncate=False)

        # 테스트 테이블 생성 및 데이터 조회
        print("\n[Test 4] 사용자 목록 조회:")
        df_users = spark.read \
            .format("jdbc") \
            .option("url", ORACLE_JDBC_URL) \
            .option("dbtable", "(SELECT username, account_status, created FROM dba_users WHERE rownum <= 10)") \
            .option("user", DB_USER) \
            .option("password", DB_PASSWORD) \
            .option("driver", ORACLE_DRIVER) \
            .load()

        df_users.show(truncate=False)

        print("\n" + "=" * 60)
        print("✅ Oracle JDBC Connection Successful!")
        print("=" * 60)

        return True

    except Exception as e:
        print(f"\n❌ Oracle Connection Failed: {e}")
        import traceback
        traceback.print_exc()
        return False


def create_sample_table(spark):
    """샘플 테이블 생성 및 데이터 삽입"""
    print("\n[Test 5] 샘플 테이블 생성:")

    try:
        # 테이블 삭제 (이미 존재하는 경우)
        drop_query = """
        BEGIN
            EXECUTE IMMEDIATE 'DROP TABLE spark_test PURGE';
        EXCEPTION
            WHEN OTHERS THEN
                IF SQLCODE != -942 THEN
                    RAISE;
                END IF;
        END;
        """

        # 테이블 생성
        create_query = """
        CREATE TABLE spark_test (
            id NUMBER PRIMARY KEY,
            name VARCHAR2(100),
            value NUMBER,
            created_date DATE DEFAULT SYSDATE
        )
        """

        # 데이터 삽입
        insert_query = """
        INSERT INTO spark_test (id, name, value)
        VALUES (1, 'Test Data 1', 100), (2, 'Test Data 2', 200), (3, 'Test Data 3', 300)
        """

        # Spark를 통하여 SQL 실행
        from pyspark.sql import SQLContext

        # 개별 SQL 실행을 위해 JDBC를 통해 직접 실행
        spark.read.format("jdbc") \
            .option("url", ORACLE_JDBC_URL) \
            .option("dbtable", f"(SELECT 1 FROM dual)") \
            .option("user", DB_USER) \
            .option("password", DB_PASSWORD) \
            .option("driver", ORACLE_DRIVER) \
            .load()

        # 테이블 생성 (JDBC를 통한 SQL 직접 실행은 제한적이므로, 데이터 조회만 수행)
        print("  Note: DDL/DML은 Oracle SQL*Plus 또는 다른 도구를 통해 실행 권장")
        print("  Spark는 주로 데이터 조회/분석에 사용")

    except Exception as e:
        print(f"  ⚠️  Table creation skipped: {e}")


def query_sample_table(spark):
    """샘플 테이블 조회"""
    print("\n[Test 6] spark_test 테이블 조회:")

    try:
        df_test = spark.read \
            .format("jdbc") \
            .option("url", ORACLE_JDBC_URL) \
            .option("dbtable", "spark_test") \
            .option("user", DB_USER) \
            .option("password", DB_PASSWORD) \
            .option("driver", ORACLE_DRIVER) \
            .load()

        df_test.show(truncate=False)
        print(f"  총 {df_test.count()} 행 조회됨")

    except Exception as e:
        print(f"  Note: 테이블이 아직 생성되지 않았을 수 있음: {e}")


def main():
    """메인 실행 함수"""
    print("🚀 Starting PySpark Oracle JDBC Connection Test")

    # Spark 세션 생성
    spark = create_spark_session()

    try:
        # Oracle 연결 테스트
        if test_oracle_connection(spark):
            # 샘플 테이블 작업
            create_sample_table(spark)
            query_sample_table(spark)

            # Spark DataFrame 예제
            print("\n[Test 7] Spark SQL 예제:")
            spark.sql("""
                SELECT
                    current_date() as spark_date,
                    current_timestamp() as spark_timestamp,
                    'Oracle JDBC 연결 성공!' as message
            """).show(truncate=False)

    finally:
        # Spark 세션 종료
        print("\n🔚 Stopping Spark Session...")
        spark.stop()
        print("✅ Done!")


if __name__ == "__main__":
    main()
